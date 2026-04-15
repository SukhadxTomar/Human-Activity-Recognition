import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_data

# Load the trained model
model = load_model("models/efficientnet_model.h5")

# Load test data
X_test, y_test = load_data(
    "Data/test/Testing_set.csv",
    "Data/test/images"
)

# Create label encoder from training data to decode predictions
X_train_temp, y_train_temp = load_data(
    "Data/train/Training_set.csv",
    "Data/train/images"
)
le = LabelEncoder()
le.fit(y_train_temp)

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = le.inverse_transform(predicted_classes)

# Check if we have true labels for evaluation
if y_test[0] is not None:  # If test labels exist
    # Encode true labels for comparison
    y_test_encoded = le.transform(y_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_encoded, predicted_classes)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, predicted_classes, target_names=le.classes_))
    
    # Show some predictions
    print("\nSample Predictions:")
    for i in range(min(10, len(X_test))):
        print(f"Image {i+1}: Predicted: {predicted_labels[i]}, Actual: {y_test[i]}")
else:  # No true labels (test data)
    print("Test data predictions (no ground truth labels available):")
    for i in range(min(10, len(X_test))):
        print(f"Image {i+1} ({X_test[i].shape}): Predicted: {predicted_labels[i]}")
    
    # Show prediction confidence for first few
    print("\nPrediction Confidence (first 5 images):")
    for i in range(min(5, len(X_test))):
        confidence = np.max(predictions[i])
        print(f"Image {i+1}: {predicted_labels[i]} ({confidence:.4f} confidence)")