"""
Grad-CAM Explainability Module
===============================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for model interpretability.
Generates visualization of which image regions influence the model's predictions.
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os


class GradCAM:
    """
    Implements Grad-CAM visualization for deep learning models.
    Shows which image regions the model focuses on for predictions.
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained Keras/TensorFlow model
            layer_name: Name of last convolutional layer (auto-detected if None)
        """
        self.model = model
        
        # Auto-detect last convolutional layer
        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        
        self.layer_name = layer_name
        self.conv_layer = model.get_layer(layer_name)
        
        print(f"✅ Grad-CAM initialized with layer: {self.layer_name}")
    
    def _find_last_conv_layer(self):
        """Auto-detect the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        raise ValueError("No convolutional layer found in model!")
    
    def compute_heatmap(self, image, pred_index=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap for an image.
        
        Args:
            image: Preprocessed image (1, 224, 224, 3)
            pred_index: Class index to compute gradient for (default: predicted class)
            eps: Small value to prevent division by zero
            
        Returns:
            Heatmap array of shape (224, 224)
        """
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[self.conv_layer.output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image, training=False)
            
            # Use predicted class if not specified
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Compute weights (average gradient)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)
        
        return heatmap.numpy()
    
    def overlay_heatmap_on_image(self, original_image, heatmap, alpha=0.5):
        """
        Overlay heatmap on original image.
        
        Args:
            original_image: Original PIL Image
            heatmap: Heatmap array (224, 224)
            alpha: Transparency of heatmap overlay (0-1)
            
        Returns:
            Overlay image as PIL Image
        """
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        
        # Normalize heatmap to 0-255
        heatmap_normalized = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply jet colormap
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Convert original image to array
        original_array = np.array(original_image, dtype=np.float32)
        
        # Blend images
        overlay = cv2.addWeighted(original_array, 1 - alpha, heatmap_colored, alpha, 0)
        overlay = overlay.astype(np.uint8)
        
        return Image.fromarray(overlay)
    
    def generate_visualization(self, original_image, preprocessed_image, 
                              pred_label, confidence, save_path=None):
        """
        Generate comprehensive Grad-CAM visualization.
        
        Args:
            original_image: Original PIL Image
            preprocessed_image: Preprocessed image (1, 224, 224, 3)
            pred_label: Predicted class label (string)
            confidence: Prediction confidence (0-1)
            save_path: Path to save visualization
            
        Returns:
            Figure with subplots
        """
        # Compute heatmap
        heatmap = self.compute_heatmap(preprocessed_image)
        
        # Create overlay
        overlay = self.overlay_heatmap_on_image(original_image, heatmap, alpha=0.6)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Grad-CAM Analysis: {pred_label} (Confidence: {confidence*100:.1f}%)', 
                     fontsize=14, fontweight='bold')
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Heatmap (Grad-CAM)', fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Activation')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay on Original', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Grad-CAM visualization saved to: {save_path}")
        
        return fig, heatmap, overlay


def create_gradcam_figure(original_image, heatmap, overlay, pred_label, confidence):
    """
    Create Grad-CAM visualization figure for Streamlit display.
    
    Args:
        original_image: Original PIL Image
        heatmap: Heatmap array
        overlay: Overlay PIL Image
        pred_label: Predicted label
        confidence: Confidence score (0-1)
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'🎯 Model Explanation - {pred_label} ({confidence*100:.1f}%)',
                 fontsize=14, fontweight='bold')
    
    # Original
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontweight='bold', fontsize=11)
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Activation Map (Grad-CAM)', fontweight='bold', fontsize=11)
    axes[1].axis('off')
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Activation Level', fontsize=9)
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Heatmap Overlay', fontweight='bold', fontsize=11)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    return fig


def get_top_activations(heatmap, num_regions=3):
    """
    Identify top activated regions in the image.
    
    Args:
        heatmap: Grad-CAM heatmap array
        num_regions: Number of top regions to identify
        
    Returns:
        List of (y, x, activation) tuples
    """
    # Find coordinates of top activations
    flat_heatmap = heatmap.flatten()
    top_indices = np.argsort(flat_heatmap)[-num_regions:][::-1]
    
    regions = []
    for idx in top_indices:
        y = idx // heatmap.shape[1]
        x = idx % heatmap.shape[1]
        activation = flat_heatmap[idx]
        regions.append((y, x, activation))
    
    return regions


def explain_prediction(model, original_image, preprocessed_image, 
                      pred_label, confidence, label_encoder=None):
    """
    Generate complete prediction explanation with Grad-CAM.
    
    Args:
        model: Trained Keras model
        original_image: Original PIL Image
        preprocessed_image: Preprocessed image (1, 224, 224, 3)
        pred_label: Predicted label (string)
        confidence: Confidence score (0-1)
        label_encoder: Optional LabelEncoder for additional info
        
    Returns:
        Dictionary with explanation components
    """
    try:
        # Initialize Grad-CAM
        grad_cam = GradCAM(model)
        
        # Compute heatmap
        heatmap = grad_cam.compute_heatmap(preprocessed_image)
        
        # Create overlay
        overlay = grad_cam.overlay_heatmap_on_image(original_image, heatmap, alpha=0.6)
        
        # Get top activations
        top_regions = get_top_activations(heatmap, num_regions=3)
        
        return {
            'grad_cam': grad_cam,
            'heatmap': heatmap,
            'overlay': overlay,
            'top_regions': top_regions,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        print(f"❌ Error generating Grad-CAM: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'grad_cam': None,
            'heatmap': None,
            'overlay': None,
            'top_regions': []
        }
