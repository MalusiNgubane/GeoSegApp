import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained model
MODEL_PATH = "height_estimation_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image: Image.Image, target_size=(256, 256)):
    """Preprocess single image for model input"""
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=-1, keepdims=True)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def display_prediction_with_gradient(image, prediction, min_elevation=0, max_elevation=267.95):
    """
    Display original image and predicted mask with height gradient
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    ax1.imshow(image, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Original Image')
    
    # Display predicted mask with color gradient
    pred_rescaled = np.interp(prediction.squeeze(), 
                             (prediction.min(), prediction.max()), 
                             (min_elevation, max_elevation))
    im = ax2.imshow(pred_rescaled, cmap='viridis', 
                    vmin=min_elevation, vmax=max_elevation)
    ax2.axis('off')
    ax2.set_title('Predicted Height Mask')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, orientation='vertical', 
                       label='Height (meters)')
    
    plt.tight_layout()
    return fig

def height_segmentation_model_page():
    st.title("Building Height Prediction from Satellite Images")
    st.write("Upload a satellite image to predict building heights and generate a height mask.")

    # Add a back button for navigation
    if st.button("Back to Main Page"):
        st.session_state.page = "main"
        st.write("Returning to the main page. Please refresh the app.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Add height range inputs
        col1, col2 = st.columns(2)
        with col1:
           min_height = st.number_input("Minimum Height (meters)", value=0.0, step=0.1)
        with col2:
           max_height = st.number_input("Maximum Height (meters)", value=267.95, step=0.1)


        # Preprocess the image for prediction
        processed_image = preprocess_image(image)

        # Button to trigger prediction
        if st.button("Predict Building Heights"):
            with st.spinner("Generating height prediction..."):
                # Get model prediction
                prediction = model.predict(processed_image)
                
                # Create visualization
                fig = display_prediction_with_gradient(
                    np.array(image), 
                    prediction, 
                    min_elevation=min_height,
                    max_elevation=max_height
                )
                
                # Display results
                st.write("### Prediction Results")
                st.pyplot(fig)
                
                # Calculate and display statistics
                pred_heights = np.interp(prediction.squeeze(), 
                                       (prediction.min(), prediction.max()), 
                                       (min_height, max_height))
                
                st.write("### Height Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Maximum Height", 
                             f"{pred_heights.max():.1f}m")
                with col2:
                    st.metric("Minimum Height", 
                             f"{pred_heights.min():.1f}m")
                with col3:
                    st.metric("Average Height", 
                             f"{pred_heights.mean():.1f}m")
                
                # Add download button for the prediction
                plt.imsave('height_prediction.png', pred_heights, 
                          cmap='viridis')
                with open('height_prediction.png', 'rb') as file:
                    st.download_button(
                        label="Download Height Prediction",
                        data=file,
                        file_name="height_prediction.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    height_segmentation_model_page()
