from data_loader import load_data
from preprocessing import preprocess_data, get_data_augmentation
from model import build_model, unfreeze_base_model
from evaluation import generate_evaluation_report

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
import pickle

print("="*70)
print("OPTIMIZED EFFICIENTNET TRAINING PIPELINE")
print("="*70)

# 1. LOAD DATA
print("\n[1/9] Loading training data...")
X, y = load_data(
    "Data/train/Training_set.csv",
    "Data/train/images"
)


# 2. PREPROCESS DATA

print("\n[2/9] Preprocessing data...")
X_train, X_val, y_train, y_val, le = preprocess_data(X, y)

# ============================================================================
# 3. ONE-HOT ENCODING
# ============================================================================
print("\n[3/9] Encoding labels...")
num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)

# Compute class weights to handle imbalanced data
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: class_weights[i] for i in range(num_classes)}
print(f"Class weights computed for balanced training")

# ============================================================================
# 4. BUILD MODEL
# ============================================================================
print("\n[4/9] Building model...")
model, base_model = build_model(num_classes)
print(f"Model built with {len(model.layers)} layers")

# ============================================================================
# 5. STAGE 1: TRAINING WITH FROZEN BASE (Transfer Learning)
# ============================================================================
print("\n[5/9] Stage 1: Training with frozen base model...")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_stage1 = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'models/best_model_stage1.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Get data augmentation
train_augmentation, val_augmentation = get_data_augmentation()

history_stage1 = model.fit(
    train_augmentation.flow(X_train, y_train_cat, batch_size=32),
    validation_data=val_augmentation.flow(X_val, y_val_cat, batch_size=32),
    epochs=30,
    callbacks=callbacks_stage1,
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================================
# 6. STAGE 2: FINE-TUNING (Unfreeze base layers)
# ============================================================================
print("\n[6/9] Stage 2: Fine-tuning base model...")
unfreeze_base_model(base_model, num_layers_to_unfreeze=50)

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_stage2 = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    ),
    ModelCheckpoint(
        'models/best_model_stage2.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history_stage2 = model.fit(
    train_augmentation.flow(X_train, y_train_cat, batch_size=16),
    validation_data=val_augmentation.flow(X_val, y_val_cat, batch_size=16),
    epochs=50,
    callbacks=callbacks_stage2,
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================================
# 7. EVALUATION
# ============================================================================
print("\n[7/9] Evaluating model...")
val_loss, val_accuracy = model.evaluate(
    X_val, y_val_cat, verbose=0
)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# ============================================================================
# 8. SAVE MODEL
# ============================================================================
print("\n[8/9] Saving model...")
os.makedirs("models", exist_ok=True)
model.save("models/efficientnet_model.h5")

# Also save label encoder for later use
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"✅ Model saved to: models/efficientnet_model.h5")
print(f"✅ Label encoder saved to: models/label_encoder.pkl")

# ============================================================================
# 9. COMPREHENSIVE EVALUATION & VISUALIZATION
# ============================================================================
print("\n[9/9] Generating comprehensive evaluation report...")
eval_results = generate_evaluation_report(
    model=model,
    X_val=X_val,
    y_val=y_val,
    history_stage1=history_stage1,
    history_stage2=history_stage2,
    label_encoder=le,
    output_dir="output"
)

print("\n" + "="*70)
print("✅ TRAINING & EVALUATION COMPLETE!")
print("="*70)
print(f"Final Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"📁 Model files saved to: models/")
print(f"📁 Evaluation results saved to: output/")
print("="*70)