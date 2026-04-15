import pandas as pd
import os
import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224  # EfficientNetB0 expects 224x224

def load_data(csv_path, img_folder):
    """Load and preprocess images from CSV and folder."""
    df = pd.read_csv(csv_path)
    
    images = []
    labels = []
    failed_count = 0
    
    for index, row in df.iterrows():
        img_name = row['filename']
        
        # Check if label column exists (for test data, it might not)
        if 'label' in df.columns:
            label = row['label']
        else:
            label = None  # No label for test data
        
        img_path = os.path.join(img_folder, img_name)
        
        try:
            img = cv2.imread(img_path)
            
            if img is None:
                failed_count += 1
                continue
            
            # Convert BGR to RGB (cv2 loads in BGR by default)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to EfficientNetB0 expected size with high quality
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply EfficientNetB0 preprocessing (normalizes to [-1, 1])
            img = preprocess_input(img)
            
            images.append(img)
            labels.append(label)
        
        except Exception as e:
            failed_count += 1
            print(f"Error loading {img_path}: {str(e)}")
            continue
    
    if failed_count > 0:
        print(f"Warning: Failed to load {failed_count} images")
    
    print(f"Successfully loaded {len(images)} images")
    return np.array(images), np.array(labels)
    
    return np.array(images), np.array(labels)