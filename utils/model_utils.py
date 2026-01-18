"""
Model utilities for attendance recognition
"""
import cv2
import numpy as np


def prepare_cell_for_prediction(cell, target_size=(64, 64)):
    """
    Prepare cell image for model prediction
    
    Args:
        cell: Input cell image
        target_size: Target size for model input
    
    Returns:
        Preprocessed cell ready for prediction
    """
    # Resize to target size
    resized = cv2.resize(cell, target_size)
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    # Reshape for model (batch, height, width, channels)
    if len(normalized.shape) == 2:  # Grayscale
        reshaped = normalized.reshape(1, *target_size, 1)
    else:  # Color
        reshaped = normalized.reshape(1, *target_size, normalized.shape[2])
    
    return reshaped


def predict_attendance_symbol(model, cell, threshold=0.5):
    """
    Predict attendance symbol from cell image
    
    Args:
        model: Trained Keras model
        cell: Cell image
        threshold: Confidence threshold
    
    Returns:
        Dictionary with prediction and confidence
    """
    try:
        # Prepare cell
        processed_cell = prepare_cell_for_prediction(cell)
        
        # Predict
        predictions = model.predict(processed_cell, verbose=0)
        
        # Get class and confidence
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        # Class labels
        classes = ["Present", "Absent", "Empty"]
        
        # Return result based on confidence
        if confidence >= threshold:
            return {
                'class': classes[class_idx],
                'confidence': float(confidence),
                'probabilities': {
                    classes[i]: float(predictions[0][i])
                    for i in range(len(classes))
                }
            }
        else:
            return {
                'class': 'Uncertain',
                'confidence': float(confidence),
                'probabilities': {
                    classes[i]: float(predictions[0][i])
                    for i in range(len(classes))
                }
            }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'class': 'Error',
            'confidence': 0.0,
            'probabilities': {}
        }


def batch_predict(model, cells, threshold=0.5):
    """
    Predict multiple cells at once (faster)
    
    Args:
        model: Trained Keras model
        cells: List of cell images
        threshold: Confidence threshold
    
    Returns:
        List of prediction dictionaries
    """
    if not cells:
        return []
    
    try:
        # Prepare all cells
        processed_cells = np.array([
            prepare_cell_for_prediction(cell)[0]
            for cell in cells
        ])
        
        # Batch prediction
        predictions = model.predict(processed_cells, verbose=0)
        
        # Process results
        classes = ["Present", "Absent", "Empty"]
        results = []
        
        for pred in predictions:
            class_idx = np.argmax(pred)
            confidence = pred[class_idx]
            
            if confidence >= threshold:
                class_name = classes[class_idx]
            else:
                class_name = 'Uncertain'
            
            results.append({
                'class': class_name,
                'confidence': float(confidence),
                'probabilities': {
                    classes[i]: float(pred[i])
                    for i in range(len(classes))
                }
            })
        
        return results
    
    except Exception as e:
        print(f"Batch prediction error: {e}")
        return [{'class': 'Error', 'confidence': 0.0, 'probabilities': {}} 
                for _ in cells]


def evaluate_model_performance(model, test_images, test_labels):
    """
    Evaluate model performance on test set
    
    Args:
        model: Trained model
        test_images: Test images
        test_labels: True labels
    
    Returns:
        Dictionary with performance metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    # Predict
    predictions = model.predict(test_images, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(true_classes, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_classes, pred_classes, average='weighted'
    )
    conf_matrix = confusion_matrix(true_classes, pred_classes)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix.tolist()
    }


def augment_training_data(images, labels, augmentation_factor=2):
    """
    Augment training data with transformations
    
    Args:
        images: Training images
        labels: Training labels
        augmentation_factor: How many augmented versions per image
    
    Returns:
        Augmented images and labels
    """
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(images, labels):
        # Add original
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # Generate augmented versions
        for _ in range(augmentation_factor):
            # Random rotation
            angle = np.random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            
            # Random noise
            noise = np.random.normal(0, 0.05, img.shape)
            noisy = np.clip(rotated + noise, 0, 1)
            
            augmented_images.append(noisy)
            augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)