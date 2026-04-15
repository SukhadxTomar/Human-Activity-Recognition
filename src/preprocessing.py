from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(X, y):
    """Preprocess data with encoding and stratified split."""
    
    # Convert labels to numbers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Stratified train/val split to maintain class distribution
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Classes: {len(le.classes_)}")
    
    return X_train, X_val, y_train, y_val, le

def get_data_augmentation():
    """Create data augmentation pipeline for training."""
    train_augmentation = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.2,
        shear_range=0.15,
        fill_mode='nearest'
    )
    
    # No augmentation for validation (only normalization via preprocess_input)
    val_augmentation = ImageDataGenerator()
    
    return train_augmentation, val_augmentation