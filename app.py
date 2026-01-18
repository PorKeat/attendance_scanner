from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
import cv2
import numpy as np
import pytesseract
import pandas as pd
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import re
import traceback

app = Flask(__name__)
app.secret_key = 'sv3-attendance-secret-key-2025'
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create folders
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load model (optional for rule-based detection)
try:
    if os.path.exists("model.h5"):
        model = load_model("model.h5")
        print("✓ Model loaded successfully")
    else:
        model = None
        print("⚠️ No model found, using rule-based detection")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

def find_and_straighten_sheet(image):
    """Detect and straighten the attendance sheet"""
    try:
        orig_h, orig_w = image.shape[:2]
        ratio = orig_h / 800.0
        img_resized = cv2.resize(image, (int(orig_w / ratio), 800))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screen_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > (img_resized.shape[0] * img_resized.shape[1] * 0.1):
                screen_cnt = approx
                break
        if screen_cnt is None:
            print("⚠️ No suitable 4-point contour found, returning original image.")
            return image
        pts = screen_cnt.reshape(4, 2) * ratio
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        print("✓ Applied perspective correction successfully.")
        return warped
    except Exception as e:
        print(f"Perspective correction failed: {e}")
        return image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_table_structure(img):
    """Detect grid lines to find table structure"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 2)
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine lines
    table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
    
    return table_structure, horizontal_lines, vertical_lines

def find_cell_grid(img):
    """Find individual cells in the table"""
    _, h_lines, v_lines = detect_table_structure(img)
    
    # Find horizontal line positions
    h_contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_positions = sorted([cv2.boundingRect(c)[1] for c in h_contours])
    
    # Find vertical line positions  
    v_contours, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_positions = sorted([cv2.boundingRect(c)[0] for c in v_contours])
    
    # Remove duplicates (lines close together)
    h_positions = [h_positions[0]] + [h for i, h in enumerate(h_positions[1:], 1) 
                                       if h - h_positions[i-1] > 20]
    v_positions = [v_positions[0]] + [v for i, v in enumerate(v_positions[1:], 1) 
                                       if v - v_positions[i-1] > 20]
    
    print(f"✓ Detected grid: {len(h_positions)} rows x {len(v_positions)} columns")
    return h_positions, v_positions

def extract_text_from_region(img, x, y, w, h, lang='eng+khm'):
    """Extract text from a specific region"""
    region = img[y:y+h, x:x+w]
    
    # Enhance for OCR
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    
    config = f'--psm 6 -l {lang}'
    text = pytesseract.image_to_string(enhanced, config=config).strip()
    
    return text

def detect_checkmark_or_cross(cell_img):
    """
    Detect checkmarks (✓) or crosses (Ø/X) in a cell using rule-based approach
    This works better than ML for handwritten symbols
    """
    if cell_img.size == 0:
        return "Empty"
    
    # Convert to grayscale
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img
    
    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Count non-zero pixels (ink)
    ink_pixels = cv2.countNonZero(binary)
    total_pixels = binary.shape[0] * binary.shape[1]
    ink_ratio = ink_pixels / total_pixels
    
    # Empty cell if very little ink
    if ink_ratio < 0.05:
        return "Empty"
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "Empty"
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Analyze shape characteristics
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    if perimeter == 0:
        return "Empty"
    
    # Circularity (for Ø detection)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Aspect ratio
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / float(h) if h > 0 else 0
    
    # Hull area ratio (for checkmark vs cross)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Decision logic
    # Ø (circle) has high circularity
    if circularity > 0.6:
        return "Absent"
    
    # Checkmark (✓) - typically elongated and less solid
    if aspect_ratio < 0.8 and solidity < 0.7 and ink_ratio > 0.1:
        return "Present"
    
    # Cross or other marks
    if ink_ratio > 0.15:
        return "Absent"
    
    return "Present" if ink_ratio > 0.08 else "Empty"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files or not request.files['file'].filename:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload an image file.', 'error')
            return redirect(url_for('index'))
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = cv2.imread(filepath)
            if img is None:
                flash('Could not read image file.', 'error')
                return redirect(url_for('index'))
            
            print(f"✓ Image loaded: {img.shape}")
            
            # Straighten image
            img = find_and_straighten_sheet(img)
            
            # Detect table grid
            h_positions, v_positions = find_cell_grid(img)
            
            if len(h_positions) < 3 or len(v_positions) < 4:
                flash('Could not detect table structure. Please ensure the image is clear.', 'error')
                return redirect(url_for('index'))
            
            # Extract header dates (assume first row after title is dates)
            dates = []
            # Skip first 3 columns (No, Khmer Name, English Name, Sex)
            for col_idx in range(4, min(len(v_positions)-1, 15)):
                x1, x2 = v_positions[col_idx], v_positions[col_idx + 1]
                y1, y2 = h_positions[0], h_positions[1]
                
                date_text = extract_text_from_region(img, x1, y1, x2-x1, y2-y1, 'eng')
                # Clean date text
                date_text = re.sub(r'[^\d/\-]', '', date_text)
                if date_text:
                    dates.append(date_text)
                else:
                    dates.append(f"Day{col_idx-3}")
            
            print(f"✓ Extracted dates: {dates}")
            
            # Extract student data (rows 2 onwards)
            records = []
            for row_idx in range(2, len(h_positions) - 1):
                y1, y2 = h_positions[row_idx], h_positions[row_idx + 1]
                
                # Extract No (column 0)
                no_text = extract_text_from_region(img, v_positions[0], y1, 
                                                   v_positions[1] - v_positions[0], y2 - y1, 'eng')
                
                # Extract English Name (column 2)
                name_text = extract_text_from_region(img, v_positions[2], y1,
                                                     v_positions[3] - v_positions[2], y2 - y1, 'eng')
                
                if not name_text or len(name_text) < 2:
                    continue  # Skip empty rows
                
                record = {'No': no_text.strip(), 'Name': name_text.strip()}
                
                # Extract attendance marks for each date
                for date_idx, date in enumerate(dates):
                    col_idx = date_idx + 4  # Start from column 4
                    
                    if col_idx >= len(v_positions) - 1:
                        break
                    
                    x1, x2 = v_positions[col_idx], v_positions[col_idx + 1]
                    
                    # Extract cell image
                    cell_img = img[y1+5:y2-5, x1+5:x2-5]  # Add padding
                    
                    # Detect mark
                    status = detect_checkmark_or_cross(cell_img)
                    
                    record[date] = status
                
                records.append(record)
                print(f"✓ Processed row {row_idx-1}: {name_text[:20]}")
            
            if not records:
                flash('Could not extract student data. Please check image quality.', 'error')
                return redirect(url_for('index'))
            
            # Create DataFrame
            df = pd.DataFrame(records)
            
            print(f"✓ Created DataFrame with {len(df)} students and {len(df.columns)} columns")
            print(f"Columns: {df.columns.tolist()}")
            
            # Save to Excel
            excel_filename = f"attendance_report_{timestamp}.xlsx"
            excel_path = os.path.join(app.config['OUTPUT_FOLDER'], excel_filename)
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Attendance', index=False)
                worksheet = writer.sheets['Attendance']
                
                # Format worksheet
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if cell.value:
                                max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 30)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                # Color code cells
                from openpyxl.styles import PatternFill
                
                green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                
                for row in worksheet.iter_rows(min_row=2, max_row=len(df)+1, 
                                               min_col=3, max_col=len(df.columns)):
                    for cell in row:
                        if cell.value == "Present":
                            cell.fill = green_fill
                        elif cell.value == "Absent":
                            cell.fill = red_fill
            
            flash(f'✅ Successfully processed attendance for {len(df)} students with {len(dates)} dates.', 'success')
            
            return send_file(
                excel_path,
                as_attachment=True,
                download_name=excel_filename,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('index'))
    
    return render_template("index.html", model_loaded=(model is not None))

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(413)
def too_large(e):
    return "File is too large. Maximum size is 16MB.", 413

if __name__ == "__main__":
    print("\n" + "="*60)
    print("📋 SV3 Attendance Recognition System - Grid Detection")
    print("="*60)
    print(f"Detection Method: Grid-based + Rule-based symbol recognition")
    print("\nSymbol Meanings:")
    print("  ✓ = Present, Ø/X = Absent, Empty = Not Marked")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)