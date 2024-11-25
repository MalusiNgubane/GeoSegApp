import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained model
MODEL_PATH = "height_estimation_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess the input image
def preprocess_image(image: Image.Image, target_size=(256, 256)):
    image = image.resize(target_size)  # Resize to model's input size
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    if len(image_array.shape) == 3:  # If RGB, convert to grayscale
        image_array = np.mean(image_array, axis=-1, keepdims=True)  # Convert to single channel
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Define the page function
def height_segmentation_model_page():
    st.title("Satellite Image Segmentation")
    st.write("Upload a satellite image, and the app will predict the building height mask.")

     # Add a back button for navigation
    if st.button("Back to Main Page"):
        st.session_state.page = "main"
        # Display a message and end execution here to simulate navigation
        st.write("Returning to the main page. Please refresh the app.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for prediction
        processed_image = preprocess_image(image)

        # Button to trigger prediction
        if st.button("Predict Height Mask"):
            with st.spinner("Predicting..."):
                prediction = model.predict(processed_image)
                predicted_mask = np.squeeze(prediction)  # Remove batch dimension

                # Display results
                st.write("**Prediction Results:**")
                fig, ax = plt.subplots(1, 2, figsize=(6, 6))
                ax[0].imshow(image)
                ax[0].set_title("Original Image")
                ax[0].axis("off")

                ax[1].imshow(predicted_mask, cmap="gray")
                ax[1].set_title("Predicted Height Mask")
                ax[1].axis("off")

                st.pyplot(fig)