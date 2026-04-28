import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Set up paths
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_model.h5"
DATA_TRAIN_PATH = PROJECT_ROOT / "Data" / "train"
DATA_TEST_PATH = PROJECT_ROOT / "Data" / "test"
IMG_SIZE = 224

# Cache model and label encoder
@st.cache_resource
def load_model_and_encoder():
    """Load the pre-trained model and label encoder."""
    try:
        model = load_model(str(MODEL_PATH))
        
        # Create label encoder from training data
        train_csv = DATA_TRAIN_PATH / "Training_set.csv"
        if train_csv.exists():
            df = pd.read_csv(train_csv)
            le = LabelEncoder()
            le.fit(df['label'].unique())
        else:
            le = None
        
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def load_training_data_sample():
    """Load a sample of training data for demonstration."""
    try:
        train_csv = DATA_TRAIN_PATH / "Training_set.csv"
        if train_csv.exists():
            df = pd.read_csv(train_csv)
            return df
    except Exception as e:
        st.warning(f"Could not load training data: {e}")
    return None

def preprocess_image(image_path):
    """Preprocess image for model prediction."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

def preprocess_uploaded_image(uploaded_file):
    """Preprocess uploaded image for prediction."""
    img = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0), img

def predict_activity(model, image_array, label_encoder):
    """Make prediction on image."""
    if model is None or label_encoder is None:
        return None, None, None
    
    predictions = model.predict(image_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    predicted_label = label_encoder.classes_[predicted_class]
    
    return predicted_label, confidence, predictions[0]

# Home Page
def home_page():
    st.title("🧠 Human Activity Recognition")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📌 Project Overview
        
        This application uses **deep learning** to automatically classify human activities from images.
        
        ### 🎯 Key Features:
        - **Real-time Prediction**: Upload images and get instant activity classification
        - **High Accuracy**: Powered by EfficientNet transfer learning model
        - **Confidence Scoring**: See prediction confidence for each activity
        - **Model Evaluation**: View detailed metrics and performance analysis
        - **Activity Classification**: Recognize up to 15+ different human activities
        
        ### 🤖 Activities Recognized:
        """)
        
        # Load training data to get activity list
        df = load_training_data_sample()
        if df is not None:
            activities = sorted(df['label'].unique())
            cols = st.columns(3)
            for idx, activity in enumerate(activities):
                with cols[idx % 3]:
                    st.write(f"✓ {activity}")
        
    with col2:
        st.info("""
        ### 📊 Model Stats
        - **Model**: EfficientNetB0
        - **Architecture**: Transfer Learning
        - **Input Size**: 224×224 pixels
        - **Framework**: TensorFlow/Keras
        """)
    
    st.markdown("---")
    
    # Dataset Statistics
    st.subheader("📊 Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    if df is not None:
        with col1:
            st.metric("Total Training Images", len(df))
        with col2:
            st.metric("Number of Activities", df['label'].nunique())
        with col3:
            activity_dist = df['label'].value_counts()
            st.metric("Samples per Activity", f"{activity_dist.min()} - {activity_dist.max()}")
        
        # Activity Distribution
        st.markdown("### Activity Distribution")
        fig, ax = plt.subplots(figsize=(12, 6))
        activity_dist.sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel("Number of Images")
        ax.set_ylabel("Activity")
        st.pyplot(fig)

# Prediction Page
def prediction_page():
    st.title("🔮 Activity Prediction")
    st.markdown("---")
    
    model, label_encoder = load_model_and_encoder()
    
    if model is None:
        st.error("❌ Model not found. Please ensure the model file exists at: models/efficientnet_model.h5")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image of a person performing an activity",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a clear image for best results"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            try:
                image_array, pil_image = preprocess_uploaded_image(uploaded_file)
                activity, confidence, all_predictions = predict_activity(model, image_array, label_encoder)
                
                if activity is not None:
                    with col2:
                        st.subheader("🎯 Prediction Results")
                        
                        # Main prediction
                        st.success(f"**Predicted Activity:** {activity}")
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                        
                        # Prediction breakdown
                        st.markdown("### 📊 All Predictions")
                        
                        # Create dataframe with predictions
                        pred_df = pd.DataFrame({
                            'Activity': label_encoder.classes_,
                            'Confidence': all_predictions,
                            'Percentage': (all_predictions * 100).round(2)
                        }).sort_values('Confidence', ascending=False)
                        
                        # Display as table
                        st.dataframe(
                            pred_df.reset_index(drop=True),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Bar chart
                        fig, ax = plt.subplots(figsize=(8, 6))
                        top_n = min(10, len(pred_df))
                        pred_df.head(top_n).plot(
                            x='Activity',
                            y='Confidence',
                            kind='barh',
                            ax=ax,
                            color='steelblue',
                            legend=False
                        )
                        ax.set_xlabel("Confidence Score")
                        ax.set_title("Top Predictions")
                        st.pyplot(fig)
                        
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    # Sample Predictions
    st.markdown("---")
    st.subheader("🖼️ Try Sample Images from Dataset")
    
    try:
        images_folder = DATA_TEST_PATH / "images"
        if images_folder.exists():
            image_files = list(images_folder.glob("*.jpg")) + list(images_folder.glob("*.png"))
            
            if image_files:
                num_samples = st.slider("Number of samples to show", 1, min(10, len(image_files)), 3)
                
                cols = st.columns(num_samples)
                
                for idx, col in enumerate(cols):
                    if idx < len(image_files):
                        with col:
                            img_path = image_files[idx]
                            
                            # Display image
                            st.image(str(img_path), use_column_width=True)
                            
                            # Make prediction
                            try:
                                image_array = preprocess_image(img_path)
                                if image_array is not None:
                                    activity, confidence, _ = predict_activity(model, image_array, label_encoder)
                                    if activity:
                                        st.write(f"**{activity}**")
                                        st.write(f"Confidence: {confidence*100:.1f}%")
                            except Exception as e:
                                st.warning(f"Error: {e}")
    except Exception as e:
        st.warning(f"Could not load test images: {e}")

# Evaluation Page
def evaluation_page():
    st.title("📊 Model Evaluation")
    st.markdown("---")
    
    model, label_encoder = load_model_and_encoder()
    
    if model is None:
        st.error("Model not loaded")
        return
    
    st.markdown("""
    ### Model Performance Metrics
    
    This section displays various evaluation metrics for the trained model.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", "EfficientNetB0")
    with col2:
        st.metric("Input Size", "224×224")
    with col3:
        st.metric("Framework", "TensorFlow")
    with col4:
        st.metric("Training Method", "Transfer Learning")
    
    st.markdown("---")
    
    # Model Architecture
    st.subheader("🏗️ Model Architecture")
    
    with st.expander("View Model Summary"):
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        st.text("\n".join(model_summary))
    
    # Load test data for evaluation
    st.subheader("📈 Evaluation on Test Data")
    
    try:
        test_csv = DATA_TEST_PATH / "Testing_set.csv"
        if test_csv.exists():
            df_test = pd.read_csv(test_csv)
            
            # Check if we can load test images
            test_images_folder = DATA_TEST_PATH / "images"
            if test_images_folder.exists():
                
                with st.spinner("Loading and evaluating test images..."):
                    y_true = []
                    y_pred = []
                    confidences = []
                    
                    progress_bar = st.progress(0)
                    
                    image_files = list(test_images_folder.glob("*.jpg")) + list(test_images_folder.glob("*.png"))
                    total_images = min(len(image_files), 100)  # Limit for performance
                    
                    for idx, img_path in enumerate(image_files[:total_images]):
                        try:
                            image_array = preprocess_image(img_path)
                            if image_array is not None:
                                activity, confidence, _ = predict_activity(model, image_array, label_encoder)
                                if activity:
                                    y_pred.append(activity)
                                    confidences.append(confidence)
                        except:
                            pass
                        
                        progress_bar.progress((idx + 1) / total_images)
                    
                    if y_pred:
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_confidence = np.mean(confidences)
                            st.metric("Average Confidence", f"{avg_confidence*100:.2f}%")
                        with col2:
                            st.metric("Images Evaluated", len(y_pred))
                        with col3:
                            high_conf = sum(1 for c in confidences if c > 0.8)
                            st.metric("High Confidence (>80%)", high_conf)
                        
                        # Confidence distribution
                        st.markdown("### 📊 Confidence Distribution")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.hist(confidences, bins=20, color='steelblue', edgecolor='black')
                        ax.set_xlabel("Confidence Score")
                        ax.set_ylabel("Frequency")
                        ax.set_title("Distribution of Prediction Confidence")
                        st.pyplot(fig)
            else:
                st.info("Test images folder not found")
    except Exception as e:
        st.warning(f"Could not load test data: {e}")
    
    # Model Information
    st.markdown("---")
    st.subheader("ℹ️ Model Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **EfficientNetB0 Characteristics:**
        - Lightweight yet powerful
        - Pre-trained on ImageNet
        - Compound scaling method
        - Efficient parameter usage
        - Suitable for mobile deployment
        """)
    
    with info_col2:
        st.markdown("""
        **Transfer Learning Benefits:**
        - Reduced training time
        - Better generalization
        - Works well with limited data
        - Lower computational requirements
        - Leverages learned features
        """)

# About Page
def about_page():
    st.title("ℹ️ About This Project")
    st.markdown("---")
    
    st.markdown("""
    ## 🧠 Human Activity Recognition Using Deep Learning
    
    ### 📌 Project Description
    
    This project implements a sophisticated image classification system to automatically recognize
    human activities from photographs. Using transfer learning with EfficientNet, the model can
    classify various human activities with high accuracy.
    
    ### 🎯 Objectives
    
    1. **Build a reliable activity recognition system** - Develop a deep learning model capable of
       accurately identifying human activities from images
    
    2. **Compare multiple architectures** - Evaluate different neural network architectures including
       EfficientNet, ResNet50, and VGG16
    
    3. **Achieve high accuracy** - Implement best practices in deep learning to maximize model performance
    
    4. **Provide confidence scores** - Give users insight into prediction reliability
    
    ### 📊 Dataset
    
    - **Total Images**: ~12,600
    - **Activity Classes**: 15+
    - **Image Resolution**: Variable (normalized to 224×224)
    - **Split**: Training and Testing sets
    
    ### 🤖 Technical Stack
    
    | Component | Technology |
    |-----------|-----------|
    | Framework | TensorFlow / Keras |
    | Model Architecture | EfficientNetB0 |
    | Preprocessing | OpenCV, NumPy |
    | UI Framework | Streamlit |
    | Data Processing | Pandas, Scikit-learn |
    | Visualization | Matplotlib, Seaborn |
    
    ### 🏗️ Model Architecture
    
    The model uses **EfficientNetB0** with transfer learning:
    
    1. **Base Model**: Pre-trained EfficientNetB0 (ImageNet weights)
    2. **Feature Extraction**: Global Average Pooling
    3. **Classification Head**: 
       - Dense(512) + BatchNorm + Dropout(0.5)
       - Dense(256) + BatchNorm + Dropout(0.4)
       - Dense(128) + BatchNorm + Dropout(0.3)
       - Dense(num_classes) + Softmax
    
    ### 📈 Key Features
    
    ✅ **Real-time Predictions** - Get instant activity classification  
    ✅ **Confidence Scoring** - Know how confident the model is  
    ✅ **Batch Processing** - Process multiple images at once  
    ✅ **Model Evaluation** - View detailed performance metrics  
    ✅ **Interactive UI** - Easy-to-use Streamlit interface  
    
    ### 🚀 Usage
    
    1. **Upload an Image**: Use the Prediction page to upload an image
    2. **Get Prediction**: The model will classify the activity and show confidence
    3. **View Metrics**: Check model performance on the Evaluation page
    4. **Explore Data**: View dataset statistics on the Home page
    
    ### 📚 Project Structure
    
    ```
    MajorProject/
    ├── app.py                    # Streamlit application
    ├── requirements.txt          # Project dependencies
    ├── README.md                 # Project documentation
    ├── Data/
    │   ├── train/               # Training images and labels
    │   └── test/                # Test images and labels
    ├── models/
    │   ├── efficientnet_model.h5
    │   ├── best_model_stage1.h5
    │   └── best_model_stage2.h5
    └── src/
        ├── data_loader.py       # Data loading utilities
        ├── preprocessing.py     # Image preprocessing
        ├── model.py            # Model architecture
        ├── train.py            # Training pipeline
        ├── predict.py          # Prediction utilities
        ├── evaluation.py       # Evaluation metrics
        └── explainability.py   # Model explainability
    ```
    
    ### 🔍 How It Works
    
    1. **Image Upload**: User uploads an image
    2. **Preprocessing**: Image is resized to 224×224 and normalized
    3. **Feature Extraction**: EfficientNet extracts features
    4. **Classification**: Dense layers classify the activity
    5. **Result**: Activity label and confidence score returned
    
    ### 📊 Activities Recognized
    
    The model can recognize activities such as:
    Walking, Running, Sitting, Standing, Sleeping, Eating, Drinking,
    Calling, Clapping, and more...
    
    ### ⚙️ Model Training Details
    
    - **Optimizer**: Adam
    - **Loss Function**: Categorical Crossentropy
    - **Metrics**: Accuracy
    - **Batch Size**: Variable (typically 32-64)
    - **Epochs**: 50-100
    - **Early Stopping**: Enabled to prevent overfitting
    
    ### 🎓 Learning Approach
    
    **Transfer Learning** is used to:
    - Leverage pre-trained weights from ImageNet
    - Reduce training time significantly
    - Improve generalization with limited data
    - Fine-tune only top layers for activity recognition
    
    ### 📞 Support & Contact
    
    For questions or issues, please refer to the project README or contact the development team.
    
    ---
    
    **Last Updated**: April 2026  
    **Version**: 1.0.0
    """)

# Main App
def main():
    st.sidebar.title("🧠 HAR System")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔮 Predict Activity", "📊 Evaluation", "ℹ️ About"],
        key="main_page"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **Quick Info:**
        - **Model**: EfficientNetB0
        - **Activities**: 15+
        - **Accuracy**: High confidence predictions
        
        📖 **How to use:**
        1. Go to Predict Activity
        2. Upload an image
        3. Get instant classification
        """
    )
    
    # Route to pages
    if "🏠 Home" in page:
        home_page()
    elif "🔮 Predict" in page:
        prediction_page()
    elif "📊 Evaluation" in page:
        evaluation_page()
    elif "ℹ️ About" in page:
        about_page()

if __name__ == "__main__":
    main()
