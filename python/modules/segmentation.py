"""
NetrAI Module 2: Retinal Structure & Lesion Segmentation
Implements classical computer vision algorithms for retinal analysis.
"""

import cv2
import numpy as np


def refine_subpixel_peak(response, x, y):
    """Refine an integer local maximum with separable quadratic interpolation."""
    values = np.asarray(response, dtype=np.float64)
    height, width = values.shape
    x = int(x)
    y = int(y)
    if x <= 0 or x >= width - 1 or y <= 0 or y >= height - 1:
        return float(x), float(y)

    center = values[y, x]
    denominator_x = values[y, x - 1] - 2.0 * center + values[y, x + 1]
    denominator_y = values[y - 1, x] - 2.0 * center + values[y + 1, x]
    offset_x = 0.0
    offset_y = 0.0
    if denominator_x < -1e-12:
        offset_x = 0.5 * (values[y, x - 1] - values[y, x + 1]) / denominator_x
    if denominator_y < -1e-12:
        offset_y = 0.5 * (values[y - 1, x] - values[y + 1, x]) / denominator_y
    return (
        float(x + np.clip(offset_x, -0.5, 0.5)),
        float(y + np.clip(offset_y, -0.5, 0.5)),
    )


def localize_optic_disc(image):
    """
    Localizes the optic disc using red channel + Circular Hough Transform.
    Returns (center_tuple, radius) with robust fallback.
    """
    h, w = image.shape[:2]

    if len(image.shape) == 3:
        red_ch = image[:, :, 0]  # RGB format: R is channel 0
    else:
        red_ch = image

    # Smooth for better circle detection
    blurred = cv2.GaussianBlur(red_ch, (9, 9), 0)

    min_dimension = min(h, w)
    min_radius = max(8, int(0.04 * min_dimension))
    max_radius = max(min_radius + 1, int(0.12 * min_dimension))

    # First estimate the disc from large-scale brightness.  This is important
    # when the disc is partially clipped by the FOV edge and HoughCircles does
    # not return it as a complete circle.
    brightness = cv2.GaussianBlur(red_ch, (0, 0), max(6, 0.03 * min_dimension))
    foreground = red_ch > 10
    brightness = brightness.copy()
    brightness[~foreground] = 0
    _, peak_value, _, brightness_peak = cv2.minMaxLoc(brightness)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dimension // 3,
        param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        # HoughCircles does not rank detections by anatomical plausibility.
        # The optic disc is the brightest candidate region in the red channel.
        scored = []
        for cx, cy, radius in circles:
            distance_to_peak = np.hypot(cx - brightness_peak[0], cy - brightness_peak[1])
            if distance_to_peak > 2 * max_radius:
                continue
            candidate_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(candidate_mask, (int(cx), int(cy)), int(radius), 255, -1)
            pixels = red_ch[candidate_mask > 0]
            score = float(pixels.mean()) if pixels.size else -1.0
            scored.append((score, int(cx), int(cy), int(radius)))
        if scored:
            _, cx, cy, radius = max(scored, key=lambda item: item[0])
            center = (cx, cy)
            return center, radius

    if peak_value > 0:
        return (int(brightness_peak[0]), int(brightness_peak[1])), int(0.075 * min_dimension)

    # Fallback: find brightest region
    _, thresh = cv2.threshold(blurred, int(np.percentile(blurred, 95)), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv2.contourArea(largest)
            radius = int(np.sqrt(area / np.pi))
            radius = int(np.clip(radius, min_radius, max_radius))
            return (cx, cy), radius

    # Ultimate fallback
    return (int(w * 0.75), int(h * 0.5)), int(min_dimension * 0.08)


def localize_fovea(green_ch, od_center, od_radius):
    """
    Locates fovea ~2.5 disc diameters temporal to optic disc.
    Searches for darkest region in the expected area.
    """
    h, w = green_ch.shape

    if od_center is None:
        return (w // 2, h // 2)

    # Determine temporal direction
    if od_center[0] < w // 2:
        direction = 1  # OD left, fovea right
    else:
        direction = -1  # OD right, fovea left

    search_x = int(od_center[0] + direction * 5 * od_radius)
    search_y = int(od_center[1])

    # Smooth and find darkest point in search window
    smoothed = cv2.GaussianBlur(green_ch, (21, 21), 10)
    win = int(2 * od_radius)
    x_min = max(0, search_x - win)
    x_max = min(w, search_x + win)
    y_min = max(0, search_y - win)
    y_max = min(h, search_y + win)

    if x_min < x_max and y_min < y_max:
        region = smoothed[y_min:y_max, x_min:x_max]
        min_idx = np.unravel_index(np.argmin(region), region.shape)
        return (x_min + min_idx[1], y_min + min_idx[0])

    return (w // 2, h // 2)


def segment_vessels(green_ch, fov_mask=None):
    """
    Segment dark vessels with multi-scale, multi-orientation line openings.

    The former Gabor/Otsu implementation was measured on DRIVE at only 0.13
    sensitivity and 0.17 Dice.  A line-opening response is less sensitive to
    global illumination and preserves elongated structures at several widths.
    The 86th response percentile is a documented operating point selected on
    the public DRIVE training masks; it still requires independent validation.
    """
    h, w = green_ch.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_ch)
    inverted = cv2.bitwise_not(enhanced)
    max_response = np.zeros_like(enhanced, dtype=np.uint8)
    scale = min(h, w) / 512.0
    lengths = []
    for reference_length in (7, 11, 15):
        length = max(3, int(round(reference_length * scale)))
        length += 1 - length % 2
        lengths.append(length)
    for length in sorted(set(lengths)):
        horizontal = np.zeros((length, length), dtype=np.uint8)
        horizontal[length // 2, :] = 1
        for angle in range(0, 180, 15):
            rotation = cv2.getRotationMatrix2D(
                (length // 2, length // 2), angle, 1.0)
            line = cv2.warpAffine(
                horizontal, rotation, (length, length),
                flags=cv2.INTER_NEAREST)
            response = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, line)
            max_response = np.maximum(max_response, response)

    valid = max_response[fov_mask > 0] if fov_mask is not None else max_response.ravel()
    valid = valid[np.isfinite(valid)]
    threshold = float(np.percentile(valid, 86)) if valid.size else 255.0
    vessel_mask = (max_response >= threshold).astype(np.uint8) * 255
    vessel_mask = cv2.morphologyEx(
        vessel_mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessel_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < max(8, int(0.00003 * h * w)):
            vessel_mask[labels == i] = 0

    if fov_mask is not None:
        vessel_mask = cv2.bitwise_and(vessel_mask, vessel_mask, mask=fov_mask)

    return vessel_mask


def detect_microaneurysms(green_ch, vessel_mask, fov_mask=None,
                          od_center=None, od_radius=None):
    """
    Detects microaneurysms using multi-scale morphological top-hat + matched filter.
    Returns (ma_mask, ma_count, ma_centroids). Centroids are floating-point
    quadratic peak estimates in analysis-image coordinates, not integer pixels.
    """
    h, w = green_ch.shape[:2]
    inverted = cv2.bitwise_not(green_ch)

    # Multi-scale top-hat
    tophat_result = np.zeros_like(inverted, dtype=np.float64)
    for r in [3, 5, 7]:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        th = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, se)
        tophat_result = np.maximum(tophat_result, th.astype(np.float64))

    # Gaussian matched filter
    sigma = 1.2
    ksize = 9
    kernel = cv2.getGaussianKernel(ksize, sigma) @ cv2.getGaussianKernel(ksize, sigma).T
    kernel = kernel / kernel.sum()
    filtered = cv2.filter2D(tophat_result, cv2.CV_64F, kernel)

    # Threshold
    mu = np.mean(filtered)
    std = np.std(filtered)
    # A stricter threshold is intentionally used here.  At 512 px the old
    # 2.5-sigma/3-pixel rule classified sensor noise and FOV-edge pixels as
    # microaneurysms on healthy APTOS images.
    thresh = mu + 3.0 * std
    ma_binary = (filtered > thresh).astype(np.uint8) * 255

    # Subtract dilated vessels
    if vessel_mask is not None:
        dilated_vessels = cv2.dilate(vessel_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        ma_binary[dilated_vessels > 0] = 0

    # Ignore border artifacts and the bright optic-disc region.
    if fov_mask is not None:
        margin = max(3, int(round(min(h, w) * 0.02)))
        kernel_size = 2 * margin + 1
        safe_fov = cv2.erode(
            fov_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )
        ma_binary[safe_fov == 0] = 0
    if od_center is not None and od_radius:
        cv2.circle(ma_binary, od_center, int(1.3 * od_radius), 0, -1)

    # Size and circularity filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ma_binary, connectivity=8)
    ma_mask = np.zeros_like(ma_binary)
    ma_centroids = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if max(5, int(0.00002 * h * w)) <= area <= max(120, int(0.0005 * h * w)):
            # Circularity check
            component = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perim = cv2.arcLength(contours[0], True)
                if perim > 0:
                    circ = 4 * np.pi * area / (perim ** 2)
                    width = stats[i, cv2.CC_STAT_WIDTH]
                    height = stats[i, cv2.CC_STAT_HEIGHT]
                    aspect = width / max(height, 1)
                    fill_ratio = area / max(width * height, 1)
                    if circ > 0.65 and 0.4 <= aspect <= 2.5 and fill_ratio >= 0.35:
                        ma_mask[labels == i] = 255
                        component_pixels = np.where(labels == i)
                        peak_index = int(np.argmax(filtered[component_pixels]))
                        peak_y = int(component_pixels[0][peak_index])
                        peak_x = int(component_pixels[1][peak_index])
                        ma_centroids.append(refine_subpixel_peak(
                            filtered, peak_x, peak_y))

    return ma_mask, len(ma_centroids), ma_centroids


def detect_hemorrhages(green_ch, vessel_mask, ma_mask, fov_mask=None,
                       od_center=None, od_radius=None):
    """
    Detects hemorrhages (dot vs blot) using morphological top-hat.
    Returns (hem_mask, hem_count, hem_stats).
    """
    h, w = green_ch.shape[:2]
    inverted = cv2.bitwise_not(green_ch)

    tophat_result = np.zeros_like(inverted, dtype=np.float64)
    for r in [10, 15]:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        th = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, se)
        tophat_result = np.maximum(tophat_result, th.astype(np.float64))

    # Normalize and threshold
    if tophat_result.max() > 0:
        norm = (tophat_result / tophat_result.max() * 255).astype(np.uint8)
        _, hem_binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        hem_binary = np.zeros_like(inverted)

    # Subtract vessels and MAs
    if vessel_mask is not None:
        dilated = cv2.dilate(vessel_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        hem_binary[dilated > 0] = 0
    if ma_mask is not None:
        hem_binary[ma_mask > 0] = 0

    if fov_mask is not None:
        margin = max(3, int(round(min(h, w) * 0.02)))
        kernel_size = 2 * margin + 1
        safe_fov = cv2.erode(
            fov_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )
        hem_binary[safe_fov == 0] = 0
    if od_center is not None and od_radius:
        cv2.circle(hem_binary, od_center, int(1.3 * od_radius), 0, -1)

    # Size filter and classify dot vs blot
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hem_binary, connectivity=8)
    hem_mask = np.zeros_like(hem_binary)
    dot_count = 0
    blot_count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if max(50, int(0.0003 * h * w)) <= area <= max(5000, int(0.02 * h * w)):
            component = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perim = cv2.arcLength(contours[0], True)
                circ = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0
                hem_mask[labels == i] = 255
                if circ > 0.6:
                    dot_count += 1
                else:
                    blot_count += 1

    return hem_mask, dot_count + blot_count, {'dot_count': dot_count, 'blot_count': blot_count}


def detect_exudates(image, green_ch, fovea_center, od_center, od_radius,
                    fov_mask=None):
    """
    Detects hard and soft exudates using LAB color space.
    Returns (ex_mask, hard_count, soft_count, fovea_dist_dd).
    """
    h, w = green_ch.shape[:2]

    # Convert to LAB
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
    else:
        L = image
        B = np.zeros_like(image)

    # OD exclusion mask
    od_mask = np.zeros((h, w), dtype=np.uint8)
    if od_center is not None and od_radius is not None:
        cv2.circle(od_mask, od_center, int(1.3 * od_radius), 255, -1)

    # Hard exudates are locally bright and yellow, not merely part of a
    # globally bright retinal region.  Local-background subtraction prevents
    # illumination gradients and FOV edges from becoming large false lesions.
    safe_fov = fov_mask
    if fov_mask is not None:
        margin = max(3, int(round(min(h, w) * 0.03)))
        kernel_size = 2 * margin + 1
        safe_fov = cv2.erode(
            fov_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )

    valid_hard = od_mask == 0
    if safe_fov is not None:
        valid_hard &= safe_fov > 0
    local_background = cv2.GaussianBlur(L, (0, 0), max(3, 0.02 * min(h, w)))
    local_contrast = L.astype(np.float32) - local_background.astype(np.float32)
    valid_values = local_contrast[valid_hard]
    contrast_threshold = (
        float(valid_values.mean() + 3.5 * valid_values.std())
        if valid_values.size else float('inf')
    )
    hard_mask = (
        (local_contrast > contrast_threshold) & (L > 130) & (B > 130) & valid_hard
    ).astype(np.uint8) * 255
    hard_mask = cv2.morphologyEx(hard_mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Remove large blobs (likely artifacts)
    num_labels, labels, stats, centroids_all = cv2.connectedComponentsWithStats(hard_mask, connectivity=8)
    clean_hard = np.zeros_like(hard_mask)
    hard_centroids = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if max(5, int(0.00002 * h * w)) <= area <= max(1000, int(0.004 * h * w)):
            clean_hard[labels == i] = 255
            hard_centroids.append(centroids_all[i])

    hard_count = len(hard_centroids)

    # Soft exudates (cotton wool spots): bright, low gradient
    grad_mag = cv2.magnitude(
        cv2.Sobel(green_ch, cv2.CV_64F, 1, 0, ksize=3),
        cv2.Sobel(green_ch, cv2.CV_64F, 0, 1, ksize=3)
    )
    grad_thresh = np.mean(grad_mag) + np.std(grad_mag)
    soft_mask = ((L > 160) & (B < 135) & (grad_mag < grad_thresh)).astype(np.uint8) * 255
    soft_mask[od_mask > 0] = 0
    if safe_fov is not None:
        soft_mask[safe_fov == 0] = 0
    soft_mask = cv2.morphologyEx(soft_mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    num_soft, _, soft_stats, _ = cv2.connectedComponentsWithStats(soft_mask, connectivity=8)
    soft_count = max(0, num_soft - 1)  # exclude background

    ex_mask = cv2.bitwise_or(clean_hard, soft_mask)

    # Distance from nearest hard exudate to fovea (in disc diameters)
    nearest_dist = -1.0
    if fovea_center is not None and hard_centroids:
        fc = np.array(fovea_center)
        dists = [np.linalg.norm(np.array(c) - fc) for c in hard_centroids]
        nearest_px = min(dists)
        if od_radius and od_radius > 0:
            nearest_dist = nearest_px / od_radius
        else:
            nearest_dist = nearest_px / 50

    return ex_mask, hard_count, soft_count, nearest_dist


def detect_neovascularization(green_ch, vessel_mask, od_center, od_radius, fov_mask=None):
    """
    Detects neovascularization via vessel density + branchpoint analysis.
    Returns (nv_mask, nv_detected).
    """
    h, w = green_ch.shape[:2]
    nv_mask = np.zeros((h, w), dtype=np.uint8)
    nv_detected = False

    if vessel_mask is None or od_center is None:
        return nv_mask, False

    # OD exclusion region
    od_exclude = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(od_exclude, od_center, int(1.5 * od_radius), 255, -1)

    # Conservative block-density screen.  A single dense block is usually a
    # normal vessel crossing, not neovascularization, so require a contiguous
    # cluster of dense blocks fully inside the retinal FOV.
    block_size = max(16, int(round(min(h, w) / 16)))
    rows = (h + block_size - 1) // block_size
    cols = (w + block_size - 1) // block_size
    densities = np.zeros((rows, cols), dtype=np.float32)
    valid = np.zeros((rows, cols), dtype=np.uint8)
    safe_fov = None
    if fov_mask is not None:
        margin = max(5, int(round(min(h, w) * 0.04)))
        kernel_size = 2 * margin + 1
        safe_fov = cv2.erode(
            fov_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )

    for row, y in enumerate(range(0, h, block_size)):
        for col, x in enumerate(range(0, w, block_size)):
            y2, x2 = min(y + block_size, h), min(x + block_size, w)
            block_area = max((y2 - y) * (x2 - x), 1)
            if safe_fov is not None:
                fov_fraction = np.sum(safe_fov[y:y2, x:x2] > 0) / block_area
                if fov_fraction < 0.90:
                    continue
            od_fraction = np.sum(od_exclude[y:y2, x:x2] > 0) / block_area
            if od_fraction > 0.25:
                continue
            valid[row, col] = 1
            densities[row, col] = np.sum(vessel_mask[y:y2, x:x2] > 0) / block_area

    valid_densities = densities[valid > 0]
    if valid_densities.size == 0:
        return nv_mask, False

    mean_density = float(valid_densities.mean())
    density_threshold = max(0.30, 3.0 * mean_density)
    candidates = ((densities > density_threshold) & (valid > 0)).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates, connectivity=8)
    min_cluster_blocks = 8
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_cluster_blocks:
            continue
        nv_detected = True
        for row, col in np.argwhere(labels == label):
            y, x = row * block_size, col * block_size
            nv_mask[y:min(y + block_size, h), x:min(x + block_size, w)] = 255

    return nv_mask, nv_detected


def extract_clinical_features(seg_results):
    """
    Extracts a 12-D normalized retinal-candidate feature vector from segmentation results.
    """
    features = np.zeros(12)

    # [0-3]: MA count per quadrant
    ma_centroids = seg_results.get('ma_centroids', [])
    od_center = seg_results.get('od_center', None)

    if od_center is not None and ma_centroids:
        q1, q2, q3, q4 = 0, 0, 0, 0
        cx, cy = od_center
        for pt in ma_centroids:
            x, y = pt
            if x >= cx and y >= cy:
                q1 += 1
            elif x < cx and y >= cy:
                q2 += 1
            elif x < cx and y < cy:
                q3 += 1
            else:
                q4 += 1
        features[0] = min(q1 / 50, 1.0)
        features[1] = min(q2 / 50, 1.0)
        features[2] = min(q3 / 50, 1.0)
        features[3] = min(q4 / 50, 1.0)
    else:
        ma_count = seg_results.get('ma_count', 0)
        features[0:4] = min(ma_count / 4 / 50, 1.0)

    # [4-7]: Hemorrhage area ratio (simplified)
    hem_mask = seg_results.get('hem_mask', None)
    if hem_mask is not None:
        total = max(hem_mask.size, 1)
        features[4:8] = np.sum(hem_mask > 0) / total

    # [8]: Hard exudate foveal distance (normalized 0-1)
    fovea_dist = seg_results.get('fovea_dist', -1)
    if fovea_dist > 0:
        features[8] = min(fovea_dist / 5.0, 1.0)
    else:
        features[8] = 1.0

    # [9]: Soft exudate presence
    features[9] = 1.0 if seg_results.get('soft_count', 0) > 0 else 0.0

    # [10]: Vessel density
    vessel_mask = seg_results.get('vessel_mask', None)
    fov_mask = seg_results.get('fov_mask', None)
    if vessel_mask is not None:
        total = max(np.sum(fov_mask > 0) if fov_mask is not None else vessel_mask.size, 1)
        features[10] = np.sum(vessel_mask > 0) / total

    # [11]: NV detected
    features[11] = 1.0 if seg_results.get('nv_detected', False) else 0.0

    return np.clip(features, 0, 1)
