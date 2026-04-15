from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

def build_model(num_classes):
    """Build optimized EfficientNetB0 model with transfer learning."""
    
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model initially
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Add dense layers with batch normalization and dropout
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    return model, base_model

def unfreeze_base_model(base_model, num_layers_to_unfreeze=50):
    """Unfreeze top layers of base model for fine-tuning."""
    base_model.trainable = True
    
    # Freeze all but the top num_layers_to_unfreeze layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    print(f"Unfroze top {num_layers_to_unfreeze} layers of base model")