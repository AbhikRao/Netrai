import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def create_circular_mask(h, w, center=None, radius=None):
    """Create a binary circular mask."""
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
    mask = dist_from_center <= radius
    return mask

def preprocess_fundus(image, target_size=512):
    """Resize preserving aspect ratio with zero-padding, extract green channel, create FOV mask."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    h, w = image.shape[:2]
    # resize preserving aspect ratio with zero-padding
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    color = [0, 0, 0]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    green_channel = padded[:, :, 1]
    gray = cv2.cvtColor(padded, cv2.COLOR_RGB2GRAY)
    _, fov_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    return padded, green_channel, fov_mask


def preprocess_for_model(image_rgb, size=512):
    """Apply the exact crop, Ben Graham enhancement, and normalization used in training."""
    if image_rgb is None or not hasattr(image_rgb, 'shape') or image_rgb.size == 0:
        raise ValueError('image_rgb must be a non-empty image')

    image = image_rgb.copy()
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('image_rgb must have shape (H, W, 3)')

    foreground = image.max(axis=2) > 7
    if foreground.any():
        rows = foreground.any(axis=1)
        cols = foreground.any(axis=0)
        row_min = int(rows.argmax())
        row_max = int(len(rows) - rows[::-1].argmax())
        col_min = int(cols.argmax())
        col_max = int(len(cols) - cols[::-1].argmax())
        image = image[row_min:row_max, col_min:col_max]

    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    image = cv2.addWeighted(
        image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128
    )
    image = image.astype(np.float32) / 255.0
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def model_space_fov_mask(image_rgb, size=512):
    """Return the retinal foreground mask in the classifier's crop/resize space."""
    if image_rgb is None or not hasattr(image_rgb, 'shape') or image_rgb.size == 0:
        raise ValueError('image_rgb must be a non-empty image')
    image = image_rgb.copy()
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    foreground = image.max(axis=2) > 7
    if foreground.any():
        rows = foreground.any(axis=1)
        cols = foreground.any(axis=0)
        row_min, row_max = int(rows.argmax()), int(len(rows) - rows[::-1].argmax())
        col_min, col_max = int(cols.argmax()), int(len(cols) - cols[::-1].argmax())
        foreground = foreground[row_min:row_max, col_min:col_max]
    resized = cv2.resize(foreground.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)

def ben_graham_preprocess(image, size=512):
    """Normalize illumination using a circular crop and Gaussian-weighted subtraction."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Crop to circle
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > 7
    image[mask == 0] = 0
    
    # resize
    image = cv2.resize(image, (size, size))
    
    # Apply Gaussian-weighted subtraction
    image_blended = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
    
    mask = np.zeros(image_blended.shape)
    cv2.circle(mask, (size//2, size//2), int(size//2 * 0.9), (1, 1, 1), -1, 8, 0)
    image_blended = image_blended * mask + 128 * (1 - mask)
    
    return image_blended.astype(np.uint8)
