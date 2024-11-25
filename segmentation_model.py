import streamlit as st
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import cv2

# Keep all your custom metrics and loss functions the same
@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(tf.cast(y_pred > 0.5, tf.float32), dtype=tf.float32)
    y_true_f = tf.cast(y_true, 'float32')
    y_pred_f = tf.keras.backend.flatten(y_pred)
    y_true_f = tf.keras.backend.flatten(y_true_f)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def combined_dice_bce_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = 1 - dice_coefficient(y_true, y_pred)
    return bce + dice

@tf.keras.utils.register_keras_serializable()
class BinaryIoU(tf.keras.metrics.Metric):
    def __init__(self, name='binary_iou', **kwargs):
        super(BinaryIoU, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return self.intersection / (self.union + tf.keras.backend.epsilon())

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

def sigmoid(x):
    """Manual sigmoid to ensure proper probability output"""
    return 1 / (1 + np.exp(-x))

def correct_model_prediction(prediction):
    """Apply sigmoid if model output seems incorrect"""
    corrected_pred = sigmoid(prediction)
    return corrected_pred

def process_prediction(prediction, threshold=0.5):
    """Enhanced prediction processing with detailed debugging"""
    # Squeeze prediction to remove batch dimension
    pred_mask = prediction.squeeze()
    
    # Print detailed prediction characteristics
    print(f"Prediction shape: {pred_mask.shape}")
    print(f"Prediction min: {pred_mask.min()}")
    print(f"Prediction max: {pred_mask.max()}")
    print(f"Prediction unique values: {np.unique(pred_mask)}")
    
    # Ensure prediction is in the correct range
    pred_mask = np.clip(pred_mask, 0, 1)
    
    # Create binary mask
    binary_mask = (pred_mask > threshold).astype(np.uint8)
    
    # Scale to 0-255 for visualization
    visible_mask = binary_mask * 255
    
    return visible_mask

class ImagePreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def normalize_image(self, image):
        """Apply the same normalization as in training"""
        image = image.astype(np.float32)
        # Standard normalization to [0, 1] range
        image = image / 255.0
        return image

    def preprocess_image(self, image):
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Print debug info
        print(f"Original image shape: {img_array.shape}")
        print(f"Original image range: [{img_array.min()}, {img_array.max()}]")
        
        # Resize image to target size
        img_resized = cv2.resize(img_array, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB if grayscale
        if len(img_resized.shape) == 2 or img_resized.shape[-1] == 1:
            img_resized = np.stack([img_resized] * 3, axis=-1)
        
        # Normalize
        img_normalized = self.normalize_image(img_resized)
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        print(f"Preprocessed image shape: {img_batch.shape}")
        print(f"Preprocessed image range: [{img_batch.min()}, {img_batch.max()}]")
        
        return img_batch

@st.cache_resource
def load_model():
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'combined_dice_bce_loss': combined_dice_bce_loss,
        'BinaryIoU': BinaryIoU
    }
    model = tf.keras.models.load_model(
        r'C:\Users\nguba\Desktop\segmodels\newmodels\final_model.h5',
        custom_objects=custom_objects
    )
    # Print model summary
    model.summary()
    return model

def segmentation_model_page():
    st.title("Satellite Image Segmentation")

    # Add a back button for navigation
    if st.button("Back to Main Page"):
        st.session_state.page = "main"
        st.write("Returning to the main page. Please refresh the app.")
        return
    
    # Create debug container
    debug_container = st.container()
    
    # Initialize model and preprocessor
    try:
        model = load_model()
        preprocessor = ImagePreprocessor()
        st.success("Model loaded successfully!")
        
        # Display model summary
        with debug_container:
            st.write("Model Summary:")
            string_io = io.StringIO()
            model.summary(print_fn=lambda x: string_io.write(x + '\n'))
            st.text(string_io.getvalue())
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    uploaded_file = st.file_uploader("Upload a satellite image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Run Segmentation"):
                with st.spinner("Processing image..."):
                    # Create debug expander
                    debug_info = st.expander("Debug Information", expanded=True)
                    
                    # Preprocess image
                    processed_input = preprocessor.preprocess_image(image)
                    
                    with debug_info:
                        st.write("Input Image Info:")
                        st.write(f"Processed input shape: {processed_input.shape}")
                        st.write(f"Input value range: [{processed_input.min():.3f}, {processed_input.max():.3f}]")
                    
                    # Run prediction with correction
                    raw_prediction = model.predict(processed_input, verbose=1)
                    prediction = correct_model_prediction(raw_prediction)
                    
                    with debug_info:
                        st.write("Prediction Info:")
                        st.write(f"Raw prediction shape: {prediction.shape}")
                        st.write(f"Raw prediction value range: [{prediction.min():.3f}, {prediction.max():.3f}]")
                        st.write(f"Unique prediction values: {np.unique(prediction)}")
                    
                    # Process prediction with multiple thresholds
                    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
                    for threshold in thresholds:
                        segmented_image = process_prediction(prediction, threshold=threshold)
                        segmented_image_pil = Image.fromarray(segmented_image)
                        st.image(segmented_image_pil, caption=f"Segmented Output (threshold={threshold})", use_column_width=True)
                        
                        # Add debug info for each threshold
                        with debug_info:
                            st.write(f"Threshold {threshold} Output Info:")
                            st.write(f"Number of positive pixels: {np.sum(segmented_image > 0)}")
                            st.write(f"Percentage of positive pixels: {(np.sum(segmented_image > 0) / segmented_image.size * 100):.2f}%")
                    
                    # Save original threshold version
                    buf = io.BytesIO()
                    segmented_image_pil.save(buf, format="PNG")
                    st.download_button(
                        label="Download Segmented Image",
                        data=buf.getvalue(),
                        file_name="segmented_output.png",
                        mime="image/png"
                    )
                    
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("Please upload an image to begin.")

if __name__ == "__main__":
    segmentation_model_page()