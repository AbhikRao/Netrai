import cv2
import numpy as np

def assess_image_quality(image):
    """
    Assess image quality and returns IQS score (0-1) and metrics dict.
    Focus: Laplacian variance on green channel
    Illumination: quadrant uniformity ratio
    FOV: coverage ratio
    IQS = 0.40*focus + 0.35*illum + 0.25*fov
    """
    if image is None or not hasattr(image, 'shape') or image.size == 0 or len(image.shape) < 2 or 0 in image.shape:
        return (0.0, {'focus': 0.0, 'illumination': 0.0, 'fov': 0.0, 'iqs': 0.0})

    h, w = image.shape[:2]
    metrics = {}
    
    if len(image.shape) == 3:
        green_ch = image[:, :, 1]
    else:
        green_ch = image
        
    # Focus: Laplacian variance on green channel
    laplacian_var = cv2.Laplacian(green_ch, cv2.CV_64F).var()
    metrics['focus'] = min(laplacian_var / max(200.0, 0.002 * h * w), 1.0)
    
    # Illumination: quadrant uniformity ratio
    h, w = green_ch.shape
    q1 = green_ch[0:h//2, 0:w//2].mean()
    q2 = green_ch[0:h//2, w//2:w].mean()
    q3 = green_ch[h//2:h, 0:w//2].mean()
    q4 = green_ch[h//2:h, w//2:w].mean()
    means = [q1, q2, q3, q4]
    illum_ratio = min(means) / (max(means) + 1e-5)
    metrics['illumination'] = illum_ratio
    
    # FOV: coverage ratio
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else green_ch
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    fov_coverage = (mask > 0).sum() / (h * w)
    metrics['fov'] = fov_coverage
    
    # IQS = 0.40*focus + 0.35*illum + 0.25*fov
    iqs = 0.40 * metrics['focus'] + 0.35 * metrics['illumination'] + 0.25 * metrics['fov']
    metrics['iqs'] = iqs
    return iqs, metrics

def enhance_fundus(image):
    """CLAHE on LAB L-channel, Gaussian background subtraction, bilateral filter denoising."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Background subtraction
    bg = cv2.GaussianBlur(cl, (0, 0), 20)
    enhanced_l = cv2.addWeighted(cl, 1.5, bg, -0.5, 0)
    
    # Bilateral filter denoising
    enhanced_l = cv2.bilateralFilter(enhanced_l, 9, 75, 75)
    
    merged = cv2.merge((enhanced_l, a, b))
    enhanced_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return enhanced_image

def generate_recapture_feedback(metrics):
    """Generate actionable operator instructions based on metrics."""
    feedback = []
    if metrics['iqs'] >= 0.7:
        feedback.append("Good quality image.")
        return feedback
        
    if metrics['focus'] < 0.5:
        feedback.append("Image is out of focus. Please refocus and ensure patient is still.")
    if metrics.get('illumination', 1.0) < 0.6:
        feedback.append("Uneven illumination detected. Check flash intensity and alignment.")
    if metrics['fov'] < 0.5:
        feedback.append("Incomplete field of view. Ensure proper alignment with pupil.")
        
    return feedback
