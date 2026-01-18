"""
Image processing utilities for attendance recognition
"""
import cv2
import numpy as np
import pytesseract


def preprocess_image(img, method='otsu'):
    """
    Preprocess image for better symbol detection
    
    Args:
        img: Input image (BGR format)
        method: Preprocessing method ('otsu', 'adaptive', 'simple')
    
    Returns:
        Binary thresholded image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if method == 'otsu':
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    else:  # simple
        _, th = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    
    return th


def enhance_image(img):
    """
    Enhance image quality for better recognition
    
    Args:
        img: Input image
    
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels
    enhanced_lab = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced


def detect_cells(binary_img, min_area=400, max_area=40000):
    """
    Detect attendance cells in the image
    
    Args:
        binary_img: Binary thresholded image
        min_area: Minimum contour area
        max_area: Maximum contour area
    
    Returns:
        List of contours representing cells
    """
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter contours by area
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            valid_contours.append(cnt)
    
    # Sort contours top to bottom, left to right
    valid_contours = sorted(
        valid_contours,
        key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0])
    )
    
    return valid_contours


def extract_cell_regions(img, contours, padding=5):
    """
    Extract cell regions from image
    
    Args:
        img: Input image
        contours: List of cell contours
        padding: Padding around cell
    
    Returns:
        List of cell images
    """
    cells = []
    h, w = img.shape[:2]
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + cw + padding)
        y2 = min(h, y + ch + padding)
        
        cell = img[y1:y2, x1:x2]
        cells.append({
            'image': cell,
            'bbox': (x, y, cw, ch),
            'position': (x, y)
        })
    
    return cells


def extract_text_ocr(img, config='--psm 6'):
    """
    Extract text from image using Tesseract OCR
    
    Args:
        img: Input image
        config: Tesseract configuration
    
    Returns:
        List of extracted text lines
    """
    try:
        # Enhance image for better OCR
        enhanced = enhance_image(img)
        
        # Extract text
        text = pytesseract.image_to_string(enhanced, config=config)
        
        # Clean and split text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        return lines
    except Exception as e:
        print(f"OCR Error: {e}")
        return []


def draw_detections(img, cells, predictions):
    """
    Draw detection boxes and predictions on image
    
    Args:
        img: Input image
        cells: List of cell dictionaries
        predictions: List of predictions
    
    Returns:
        Image with drawings
    """
    result = img.copy()
    
    for cell, pred in zip(cells, predictions):
        x, y, w, h = cell['bbox']
        
        # Choose color based on prediction
        if pred == 'Present':
            color = (0, 255, 0)  # Green
        elif pred == 'Absent':
            color = (0, 0, 255)  # Red
        else:
            color = (128, 128, 128)  # Gray
        
        # Draw rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        cv2.putText(result, pred, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result


def deskew_image(img):
    """
    Deskew (rotate) a skewed image
    
    Args:
        img: Input image
    
    Returns:
        Deskewed image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    
    # Detect angle
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    
    return rotated