import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import os
from scipy.ndimage import map_coordinates, gaussian_filter

# --- Augmentation Functions ---

def apply_elastic_transform(image, alpha, sigma, random_state=None):
    """
    Apply elastic transformation on an image as described in [Simard et al., 2003].
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def apply_motion_blur(image, size):
    """Apply motion blur to an image."""
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(image, -1, kernel)

def apply_perspective_transform(image, magnitude=0.1):
    """Apply a random perspective transform."""
    h, w = image.shape
    # Define original and new points
    pts1 = np.float32([[0,0],[w-1,0],[0,h-1],[w-1,h-1]])
    # Get random offsets for each corner
    tl_x, tl_y = w * magnitude * np.random.uniform(-1, 1), h * magnitude * np.random.uniform(-1, 1)
    tr_x, tr_y = w * magnitude * np.random.uniform(-1, 1), h * magnitude * np.random.uniform(-1, 1)
    bl_x, bl_y = w * magnitude * np.random.uniform(-1, 1), h * magnitude * np.random.uniform(-1, 1)
    br_x, br_y = w * magnitude * np.random.uniform(-1, 1), h * magnitude * np.random.uniform(-1, 1)
    
    pts2 = np.float32([
        [tl_x, tl_y],                  # Top-left
        [w - 1 + tr_x, tr_y],          # Top-right
        [bl_x, h - 1 + bl_y],          # Bottom-left
        [w - 1 + br_x, h - 1 + br_y]   # Bottom-right
    ])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(image,M,(w,h), borderValue=255)

def apply_occlusion(image, max_rects=1, min_size=5, max_size=15):
    """Add random occlusions (black rectangles) to the image."""
    img_occ = image.copy()
    for _ in range(np.random.randint(1, max_rects + 1)):
        x1 = np.random.randint(0, image.shape[1])
        y1 = np.random.randint(0, image.shape[0])
        occ_w = np.random.randint(min_size, max_size)
        occ_h = np.random.randint(min_size, max_size)
        x2 = np.clip(x1 + occ_w, 0, image.shape[1])
        y2 = np.clip(y1 + occ_h, 0, image.shape[0])
        color = np.random.randint(0, 50) # Dark gray occlusion
        cv2.rectangle(img_occ, (x1, y1), (x2, y2), color, -1)
    return img_occ

# --- Data Generation ---

def create_realistic_training_data(samples_per_class=1000):
    """
    Create training data based on actual attendance sheet symbols with advanced augmentations.
    0: Checkmark (✓) - PRESENT
    1: Letter (A) - ABSENT
    2: Empty - No mark
    """
    X = []
    y = []
    
    for label in range(3):
        for _ in range(samples_per_class):
            img = np.ones((64, 64), dtype=np.uint8) * 255
            
            if label == 0:  # Checkmark ✓ = PRESENT
                pts1 = np.array([[18, 32], [24, 40], [22, 42], [16, 34]], np.int32)
                pts2 = np.array([[24, 40], [48, 16], [50, 18], [26, 42]], np.int32)
                thickness = np.random.choice([2, 3])
                cv2.fillPoly(img, [pts1], 0)
                cv2.fillPoly(img, [pts2], 0)
                if np.random.random() > 0.7:
                    cv2.polylines(img, [pts1], False, 50, 1)
                    cv2.polylines(img, [pts2], False, 50, 1)
                
            elif label == 1:  # Letter A = ABSENT
                thickness = 2
                cv2.line(img, (32, 45), (20, 20), 0, thickness)
                cv2.line(img, (32, 45), (44, 20), 0, thickness)
                cv2.line(img, (26, 35), (38, 35), 0, thickness)
                
            else:  # Empty cell
                cv2.line(img, (0, 32), (64, 32), 220, 1)
                cv2.line(img, (32, 0), (32, 64), 220, 1)
            
            # --- Basic Augmentations ---
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((32, 32), angle, 1)
            img = cv2.warpAffine(img, M, (64, 64), borderValue=255)
            
            noise = np.random.normal(0, 8, (64, 64))
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            shadow = np.random.randint(-20, -5)
            img = np.clip(img + shadow, 0, 255).astype(np.uint8)
            
            brightness = np.random.randint(-15, 15)
            img = np.clip(img + brightness, 0, 255).astype(np.uint8)

            # --- Advanced Augmentations ---
            if np.random.random() > 0.5:
                img = apply_perspective_transform(img, magnitude=0.08)

            if np.random.random() > 0.7:
                img = apply_elastic_transform(img, alpha=np.random.randint(30, 40), sigma=np.random.randint(4, 6))
            
            if np.random.random() > 0.5:
                blur_type = np.random.choice(['gaussian', 'motion'])
                if blur_type == 'gaussian':
                    kernel_size = np.random.choice([3, 5])
                    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                else:
                    motion_kernel_size = np.random.choice([5, 7, 9])
                    img = apply_motion_blur(img, motion_kernel_size)
            
            if np.random.random() > 0.85 and label != 2: # Avoid occluding empty cells too much
                img = apply_occlusion(img)

            X.append(img)
            y.append(label)
    
    return np.array(X), np.array(y)

def load_images_from_folder(folder):
    """
    Load images from a folder with subdirectories as class labels.
    """
    X = []
    y = []
    class_map = {"present": 0, "absent": 1, "not_marked": 2}
    
    if not os.path.exists(folder):
        print(f"Directory not found: {folder}")
        return None, None

    print(f"🖼️ Loading images from '{folder}'...")
    for class_name, class_idx in class_map.items():
        class_path = os.path.join(folder, class_name)
        if not os.path.exists(class_path):
            print(f"   - Warning: Class folder not found, skipping: '{class_path}'")
            continue
        
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"   - Warning: No images found in '{class_path}'")
            continue

        print(f"   - Loading class '{class_name}': {len(image_files)} images")
        for filename in image_files:
            try:
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"     - Warning: Could not read image, skipping: '{img_path}'")
                    continue
                
                if np.mean(img) < 128:
                    img = cv2.bitwise_not(img)

                img = cv2.resize(img, (64, 64))
                
                X.append(img)
                y.append(class_idx)
            except Exception as e:
                print(f"     - Error processing image '{img_path}': {e}")

    if not X:
        return None, None
        
    return np.array(X), np.array(y)


# --- Main Training Script ---

print("=" * 60)
print("Training Attendance Recognition Model")
print("Optimized for SV3 (A) Class Attendance Sheet")
print("=" * 60)

print("\n📊 Symbol Meanings:")
print("   - present: ✓ (Checkmark)")
print("   - absent: A (Letter)")
print("   - not_marked: Empty cell\n")

# --- Data Loading ---
DATASET_DIR = 'dataset'
X, y = None, None

if os.path.exists(DATASET_DIR) and any(os.scandir(DATASET_DIR)):
    print(f"\nFound '{DATASET_DIR}' directory, loading images from folder.")
    X, y = load_images_from_folder(DATASET_DIR)

if X is None or y is None:
    if os.path.exists(DATASET_DIR):
        print(f"\n⚠️  '{DATASET_DIR}' is empty or invalid.")
    else:
        print(f"\n⚠️  '{DATASET_DIR}' not found.")
    
    print("Falling back to generating synthetic data with augmentations...")
    X, y = create_realistic_training_data(samples_per_class=2000) # Increased samples for more variety
    print("🔄 Generating realistic training data...")
else:
    print("\n✅ Successfully loaded data from dataset folder.")

# --- Preprocessing ---
print("\n⚙️  Preprocessing data...")

# --- Validate dataset size before splitting ---
if y is not None and len(y) > 0 and y.ndim == 1:
    import math
    
    unique_labels, counts = np.unique(y, return_counts=True)
    n_classes = len(unique_labels)
    total_samples = len(y)
    test_set_fraction = 0.2
    
    class_map_rev = {0: "present", 1: "absent", 2: "not_marked"}

    if n_classes < 2:
        print("\n❌ FATAL ERROR: Your dataset must contain images for at least TWO different classes.")
        exit()

    min_samples_per_class = 2
    for label, count in zip(unique_labels, counts):
        if count < min_samples_per_class:
            class_name = class_map_rev.get(label, f"Class {label}")
            print(f"\n❌ FATAL ERROR: Not enough images for class '{class_name}'. At least {min_samples_per_class} are required.")
            exit()
            
    test_samples_count = math.ceil(test_set_fraction * total_samples)
    if test_samples_count < n_classes:
        required_total_samples = 5 * (n_classes - 1) + 1
        more_images_needed = required_total_samples - total_samples
        print(f"\n❌ FATAL ERROR: Your dataset is too small to be split properly ({total_samples} images, {n_classes} classes).")
        if more_images_needed > 0:
            print(f"Please add at least {more_images_needed} more images to your '{DATASET_DIR}' subfolders.")
        exit()

# --- Model Training ---
X = X / 255.0
X = X.reshape(-1, 64, 64, 1)

y = to_categorical(y, 3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Validation samples: {len(X_test)}")

print("\n🧠 Building enhanced CNN architecture...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1), padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Summary:")
model.summary()

early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=7, # Increased patience
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6,
    verbose=1
)

print("\n🚀 Training started...")
print("This may take 5-15 minutes depending on your hardware and dataset size...\n")

history = model.fit(
    X_train, y_train,
    epochs=40, # Increased epochs for augmented data
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n📊 Final Evaluation:")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✓ Test Loss: {test_loss:.4f}")
print(f"✓ Test Accuracy: {test_acc*100:.2f}%")

model.save("model.h5")
print("\n💾 Model saved as 'model.h5'")

for folder in ['uploads', 'outputs', 'templates', 'static']:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"✓ Created '{folder}' folder")

print("\n" + "=" * 60)
print("🎉 Training Complete!")
print(f"📈 Final Accuracy: {test_acc*100:.2f}%")
print("\n✅ Your model is ready to process attendance sheets!")
print("\nNext steps:")
print("  1. Run: python app.py")
print("  2. Go to http://127.0.0.1:5000 in your browser")
print("=" * 60)