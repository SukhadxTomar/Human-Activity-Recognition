"""
Human Activity Recognition System using EfficientNet
========================================================

This is a production-level Streamlit application for predicting human activities
from images using a pre-trained EfficientNet model.

Requirements:
    - tensorflow >= 2.10.0
    - streamlit >= 1.28.0
    - numpy >= 1.20.0
    - pillow >= 8.0.0
    - scikit-learn >= 1.0.0

Installation:
    pip install tensorflow streamlit numpy pillow scikit-learn

Running the app:
    streamlit run app.py

Project Structure:
    - models/efficientnet_model.h5 (trained EfficientNet model)
    - app.py (this file)
    - images/ (folder for uploaded images)
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional
import warnings
import pickle
import pandas as pd
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from explainability import explain_prediction, create_gradcam_figure

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# EfficientNet input specifications
MODEL_INPUT_SIZE = 224
MODEL_PATH = "models/efficientnet_model.h5"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
TRAINING_CSV_PATH = "Data/train/Training_set.csv"

# Load class names from Training_set.csv
def load_class_names_from_csv(csv_path: str):
    """Load unique class names from the training CSV file."""
    try:
        df = pd.read_csv(csv_path)
        # Get unique labels and sort them
        class_names = sorted(df['label'].unique().tolist())
        return class_names
    except Exception as e:
        st.error(f"Error loading labels from CSV: {str(e)}")
        return []

CLASS_NAMES = load_class_names_from_csv(TRAINING_CSV_PATH)

# Confidence threshold for prediction certainty
CONFIDENCE_THRESHOLD = 0.60

# Page configuration
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CSS STYLING - Professional Black & Green Theme
# ============================================================================

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0e27;
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f1535;
        border-right: 2px solid #00d084;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #00d084;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0f1535 0%, #1a1f4b 100%);
        border: 2px solid #00d084;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 16px rgba(0, 208, 132, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d084;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 208, 132, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d084 0%, #00a366 100%);
        color: #0a0e27 !important;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 208, 132, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 208, 132, 0.5);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #00d084 !important;
        border-radius: 8px;
    }
    
    /* Success message */
    .success-box {
        background: rgba(0, 208, 132, 0.1);
        border-left: 4px solid #00d084;
        padding: 15px;
        border-radius: 6px;
        color: #00d084;
    }
    
    /* Warning message */
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 6px;
        color: #ffc107;
    }
    
    /* Error message */
    .error-box {
        background: rgba(231, 76, 60, 0.1);
        border-left: 4px solid #e74c3c;
        padding: 15px;
        border-radius: 6px;
        color: #e74c3c;
    }
    
    /* Confidence score styling */
    .confidence-score {
        font-size: 2.5em;
        font-weight: 700;
        text-align: center;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .confidence-high {
        background: linear-gradient(135deg, rgba(0, 208, 132, 0.2) 0%, rgba(0, 208, 132, 0.1) 100%);
        border: 2px solid #00d084;
        color: #00d084;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.2) 0%, rgba(255, 193, 7, 0.1) 100%);
        border: 2px solid #ffc107;
        color: #ffc107;
    }
    
    /* Container styling */
    .container {
        background: linear-gradient(135deg, #0f1535 0%, #1a1f4b 100%);
        border: 1px solid #00d084;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Text styling */
    .subtitle {
        color: #00d084;
        font-weight: 600;
        font-size: 1.1em;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* Info text */
    .info-text {
        color: #b0b0b0;
        font-size: 0.95em;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING & CACHING
# ============================================================================

@st.cache_resource
def load_model(model_path: str):
    """
    Load the pre-trained EfficientNet model from disk.
    Uses Streamlit caching to avoid reloading on every interaction.
    
    Args:
        model_path: Path to the trained model file
        
    Returns:
        Loaded Keras model
        
    Raises:
        FileNotFoundError: If model file not found
        Exception: If model loading fails
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please ensure the model file exists."
            )
        
        # Suppress TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(model_path)
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise


@st.cache_resource
def load_label_encoder():
    """
    Load the pre-trained label encoder from disk.
    Uses Streamlit caching to avoid reloading on every interaction.
    
    Returns:
        Fitted LabelEncoder object
        
    Raises:
        FileNotFoundError: If encoder file not found
        Exception: If loading fails
    """
    try:
        if os.path.exists(LABEL_ENCODER_PATH):
            with open(LABEL_ENCODER_PATH, 'rb') as f:
                le = pickle.load(f)
            return le
        else:
            # Fallback: create encoder from CSV labels
            encoder = LabelEncoder()
            encoder.fit(CLASS_NAMES)
            return encoder
    
    except Exception as e:
        st.warning(f"Could not load saved encoder: {str(e)}")
        encoder = LabelEncoder()
        encoder.fit(CLASS_NAMES)
        return encoder


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for EfficientNet model prediction.
    
    Steps:
    1. Convert to RGB (if needed)
    2. Resize to 224x224 (EfficientNet standard input)
    3. Convert to numpy array
    4. Apply EfficientNet preprocessing (normalizes to [-1, 1])
    5. Expand dimensions for batch processing
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed numpy array ready for model prediction
    """
    # Convert to RGB if image has alpha channel or is grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to EfficientNet standard input size with high quality
    image = image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.LANCZOS)
    
    # Convert PIL Image to numpy array
    image_array = np.array(image, dtype=np.float32)
    
    # Apply EfficientNet preprocessing (normalizes to [-1, 1])
    # This is consistent with training preprocessing
    image_array = preprocess_input(image_array)
    
    # Expand dimensions for batch processing: (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_activity(
    model, 
    image: Image.Image, 
    label_encoder: LabelEncoder
) -> Tuple[str, float, bool]:
    """
    Perform prediction on the input image.
    
    Args:
        model: Loaded Keras model
        image: PIL Image object
        label_encoder: Fitted LabelEncoder for converting indices to labels
        
    Returns:
        Tuple containing:
        - predicted_label: String name of predicted activity
        - confidence: Float confidence score (0-1)
        - is_confident: Boolean indicating if prediction is confident
    """
    try:
        # Preprocess image
        preprocessed_image = preprocess_image(image)
        
        # Get predictions from model
        # model.predict returns array of shape (1, num_classes)
        predictions = model.predict(preprocessed_image, verbose=0)
        
        # Get the confidence score (max probability)
        confidence = float(np.max(predictions))
        
        # Get the class index with highest probability
        predicted_class_idx = np.argmax(predictions[0])
        
        # Convert class index to label using LabelEncoder
        predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Determine if prediction is confident
        is_confident = confidence >= CONFIDENCE_THRESHOLD
        
        return predicted_label, confidence, is_confident
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        raise


# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_header():
    """Display the main header with title and description."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <h1 style='margin-bottom: 0;'>🎯 Human Activity Recognition System</h1>
        """, unsafe_allow_html=True)
        st.markdown("""
        <p class='info-text'>
        Powered by EfficientNet - Detect human activities from images with high accuracy
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: right; padding: 20px;'>
            <p style='color: #00d084; font-weight: 700; font-size: 1.2em;'>⚡ AI Powered</p>
            <p style='color: #b0b0b0; font-size: 0.9em;'>EfficientNet v0</p>
        </div>
        """, unsafe_allow_html=True)


def display_instructions():
    """Display usage instructions and supported formats."""
    with st.expander("📋 How to Use", expanded=False):
        st.markdown("""
        ### Getting Started
        
        1. **Upload an Image**
           - Click the uploader below
           - Choose a JPG or PNG image
           - Image should contain a clear human activity
        
        2. **Review the Image**
           - Verify the uploaded image is correct
           - Check image quality and lighting
        
        3. **Get Prediction**
           - Click the "Predict Activity" button
           - Wait for the model to analyze the image
           - View results with confidence score
        
        ### Supported Activities
        """)
        
        # Display supported activities in columns
        cols = st.columns(4)
        for idx, activity in enumerate(CLASS_NAMES):
            with cols[idx % 4]:
                st.markdown(f"✓ {activity}")
        
        st.markdown("""
        ### Tips for Best Results
        - Use clear, well-lit images
        - Ensure the person in the image is performing the activity clearly
        - Avoid partial/blurred images
        - Confidence > 60% indicates a reliable prediction
        """)


def display_upload_section():
    """Display the file upload component."""
    st.markdown("<p class='subtitle'>📸 Upload Image</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        label="Choose an image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing human activity"
    )
    
    return uploaded_file


def display_image_preview(uploaded_file):
    """Display the uploaded image."""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<p class='subtitle'>👁️ Uploaded Image</p>", unsafe_allow_html=True)
            st.image(
                image,
                use_column_width=True,
                caption="Preview of uploaded image"
            )
        
        with col2:
            st.markdown("<p class='subtitle'>📊 Image Details</p>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='container'>
                <p><strong>Format:</strong> {uploaded_file.type}</p>
                <p><strong>Size:</strong> {uploaded_file.size / 1024:.2f} KB</p>
                <p><strong>Dimensions:</strong> {image.size[0]}×{image.size[1]} px</p>
                <p><strong>Mode:</strong> {image.mode}</p>
            </div>
            """, unsafe_allow_html=True)
        
        return image
    
    return None


def display_prediction_results(predicted_label: str, confidence: float, is_confident: bool):
    """
    Display prediction results with proper styling based on confidence level.
    
    Args:
        predicted_label: The predicted activity label
        confidence: Confidence score (0-1)
        is_confident: Whether prediction is confident enough
    """
    st.markdown("<p class='subtitle'>🎯 Prediction Results</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display predicted activity
        st.markdown("""
        <div class='container'>
            <p style='color: #b0b0b0; margin: 0;'>Predicted Activity</p>
            <p style='color: #00d084; font-size: 2em; font-weight: 700; margin: 10px 0; text-transform: uppercase;'>
                {activity}
            </p>
        </div>
        """.format(activity=predicted_label), unsafe_allow_html=True)
    
    with col2:
        # Display confidence score with color coding
        confidence_pct = confidence * 100
        
        if is_confident:
            confidence_class = "confidence-high"
            status_icon = "✅"
            status_text = "High Confidence"
        else:
            confidence_class = "confidence-low"
            status_icon = "⚠️"
            status_text = "Uncertain Prediction"
        
        st.markdown(f"""
        <div class='{confidence_class} confidence-score'>
            {confidence_pct:.1f}%
        </div>
        <div style='text-align: center; color: #b0b0b0;'>
            <p style='margin: 0;'>{status_icon} {status_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display confidence interpretation
    if is_confident:
        st.markdown("""
        <div class='success-box'>
            ✅ <strong>Confident Prediction</strong> - The model is confident about this prediction.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warning-box'>
            ⚠️ <strong>Uncertain Prediction</strong> - The model's confidence is below the threshold.
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR INFORMATION
# ============================================================================

def display_sidebar():
    """Display sidebar information and settings."""
    with st.sidebar:
        st.markdown("<p style='color: #00d084; font-weight: 700; font-size: 1.3em;'>⚙️ Settings</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("<p class='subtitle'>Model Configuration</p>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='container'>
            <p><strong>Model:</strong> EfficientNet</p>
            <p><strong>Input Size:</strong> {MODEL_INPUT_SIZE}×{MODEL_INPUT_SIZE}</p>
            <p><strong>Classes:</strong> {len(CLASS_NAMES)}</p>
            <p><strong>Confidence Threshold:</strong> {CONFIDENCE_THRESHOLD * 100:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p class='subtitle'>Supported Activities</p>", unsafe_allow_html=True)
        for idx, activity in enumerate(CLASS_NAMES, 1):
            st.markdown(f"**{idx}.** {activity}", help=f"Class index: {idx-1}")
        
        st.markdown("---")
        
        st.markdown("<p class='subtitle'>📚 Information</p>", unsafe_allow_html=True)
        st.markdown("""
        - **Model Framework:** TensorFlow/Keras
        - **Input Format:** RGB Images
        - **Normalization:** [0, 1] range
        - **Color Scheme:** Professional Dark Theme
        
        ---
        
        ### Quick Start
        ```bash
        streamlit run app.py
        ```
        
        ### Requirements
        - TensorFlow 2.10+
        - Streamlit 1.28+
        - NumPy, Pillow, scikit-learn
        """)


# ============================================================================
# EXPLAINABILITY VISUALIZATION
# ============================================================================

import matplotlib.pyplot as plt


def display_explainability(model, original_image, preprocessed_image, 
                            predicted_label, confidence, label_encoder):
    """
    Display Grad-CAM explainability visualization.
    Shows which parts of the image the model focused on for the prediction.
    
    Args:
        model: Trained Keras model
        original_image: Original PIL Image
        preprocessed_image: Preprocessed image (1, 224, 224, 3)
        predicted_label: Predicted activity label
        confidence: Confidence score (0-1)
        label_encoder: LabelEncoder for class info
    """
    st.markdown("<p class='subtitle'>🔍 Model Explainability (Grad-CAM)</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-text'>
    Grad-CAM shows which regions of the image influenced the model's prediction. 
    Brighter/warmer areas = higher importance for the prediction.
    </div>
    """, unsafe_allow_html=True)
    
    # Generate Grad-CAM visualization
    with st.spinner("🔍 Generating Grad-CAM explanation..."):
        try:
            explanation = explain_prediction(
                model=model,
                original_image=original_image,
                preprocessed_image=preprocessed_image,
                pred_label=predicted_label,
                confidence=confidence,
                label_encoder=label_encoder
            )
            
            if explanation['success']:
                # Display Grad-CAM figure
                fig = create_gradcam_figure(
                    original_image,
                    explanation['heatmap'],
                    explanation['overlay'],
                    predicted_label,
                    confidence
                )
                
                st.pyplot(fig, use_container_width=True)
                
                # Display interpretation
                st.markdown("""
                <div class='success-box'>
                <strong>📊 Interpretation:</strong><br/>
                • <strong>Left:</strong> Your original uploaded image<br/>
                • <strong>Center:</strong> Heat map showing model focus areas<br/>
                • <strong>Right:</strong> Heat map overlaid on original image<br/>
                <br/>
                <strong>Red/Warm colors</strong> = High activation | <strong>Blue/Cool colors</strong> = Low activation
                </div>
                """, unsafe_allow_html=True)
                
                # Close figure to free memory
                plt.close(fig)
            
            else:
                st.warning(f"⚠️ Could not generate Grad-CAM: {explanation['error']}")
        
        except Exception as e:
            st.error(f"❌ Error generating explanation: {str(e)}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    st.markdown("---")
    
    # Display instructions
    display_instructions()
    
    st.markdown("---")
    
    # File upload section
    uploaded_file = display_upload_section()
    
    if uploaded_file is not None:
        # Display image preview
        image = display_image_preview(uploaded_file)
        
        st.markdown("---")
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            predict_button = st.button(
                "🚀 Predict Activity",
                use_container_width=True,
                help="Click to analyze the image and predict the human activity"
            )
        
        if predict_button and image is not None:
            with st.spinner("🔄 Analyzing image... Please wait."):
                try:
                    # Load model and encoder
                    model = load_model(MODEL_PATH)
                    label_encoder = load_label_encoder()
                    
                    # Get prediction
                    predicted_label, confidence, is_confident = predict_activity(
                        model,
                        image,
                        label_encoder
                    )
                    
                    # Display results
                    display_prediction_results(predicted_label, confidence, is_confident)
                    
                    # Display explainability visualization (Grad-CAM)
                    st.markdown("---")
                    
                    # Prepare preprocessed image for Grad-CAM
                    preprocessed_for_gradcam = preprocess_image(image)
                    
                    # Display Grad-CAM explanation
                    display_explainability(
                        model=model,
                        original_image=image,
                        preprocessed_image=preprocessed_for_gradcam,
                        predicted_label=predicted_label,
                        confidence=confidence,
                        label_encoder=label_encoder
                    )
                    
                except FileNotFoundError as e:
                    st.markdown(f"""
                    <div class='error-box'>
                    ❌ <strong>Model Not Found</strong><br/>
                    {str(e)}<br/>
                    Please ensure the model file exists at: {MODEL_PATH}
                    </div>
                    """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.markdown(f"""
                    <div class='error-box'>
                    ❌ <strong>Prediction Error</strong><br/>
                    {str(e)}<br/>
                    Please try again with a different image.
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        # Empty state
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 40px; color: #b0b0b0;'>
                <p style='font-size: 3em; margin: 20px 0;'>📷</p>
                <p style='font-size: 1.2em; font-weight: 600; margin: 10px 0;'>Upload an image to get started</p>
                <p style='font-style: italic; margin: 10px 0;'>Supported formats: JPG, PNG</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666666; padding: 20px; font-size: 0.85em;'>
        <p>Human Activity Recognition System v1.0</p>
        <p>Powered by EfficientNet | Built with Streamlit & TensorFlow</p>
        <p>© 2024 | Professional Deep Learning Application</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()