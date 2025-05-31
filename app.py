import streamlit as st
import tensorflow as tf
from PIL import Image # For image manipulation
import numpy as np    # For numerical operations
import io             # For handling file uploads
import cv2            # For OpenCV, used for camera input (bonus)

st.set_page_config(page_title="InspectorsAlly - Bolt Quality Check", page_icon=":wrench:")

st.title("InspectorsAlly - Bolt Quality Control")
st.caption("Boost Your Quality Control with AI-Powered Bolt Inspection")
st.write("Upload an image of a bolt or use your camera to classify it as Good / Anomaly.")

# Load the Keras model
@st.cache_resource # Use st.cache_resource to load the model only once
def load_model():
    model = tf.keras.models.load_model('./model/keras_model.h5') # Adjust path if different
    return model

model = load_model()

# Load the labels
with open('./model/labels.txt', 'r') as f: # Adjust path if different
    class_names = [line.strip().split(' ', 1)[1] for line in f] # To get just the name, not '0 Normal_Bolt'


def preprocess_image(image):
    # Resize the image to 224x224 pixels
    image = image.resize((224, 224))
    # Convert the image to a NumPy array
    image_array = np.asarray(image)
    # Normalize the image (Teachable Machine's specific normalization)
    # Pixel values usually in [0, 255] are scaled to [-1, 1]
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Add a batch dimension (model expects a batch of images)
    return np.expand_dims(normalized_image_array, axis=0)

def predict_bolt_anomaly(image):
    data = preprocess_image(image)
    prediction = model.predict(data)
    # Get the index of the class with the highest confidence
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence_score = prediction[0][predicted_class_index] * 100 # Convert to percentage

    return predicted_class_name, confidence_score

input_method = st.radio("Choose Input Method:", ("File Uploader", "Camera Input (Bonus)"))

uploaded_file_img = None
camera_file_img = None

if input_method == "File Uploader":
    uploaded_file_img = st.file_uploader(
        "Upload an image of a bolt",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a bolt for classification."
    )
elif input_method == "Camera Input (Bonus)":
    camera_file_img = st.camera_input("Take a picture of a bolt")

# This button will trigger the prediction
submit_button = st.button("Analyze Bolt")

if submit_button:
    image_to_process = None
    if uploaded_file_img is not None:
        image_to_process = Image.open(io.BytesIO(uploaded_file_img.read()))
    elif camera_file_img is not None:
        image_to_process = Image.open(io.BytesIO(camera_file_img.read()))

    if image_to_process is not None:
        st.subheader("Input Image:")
        st.image(image_to_process, caption='Image for Analysis', use_column_width=True)

        predicted_class, confidence = predict_bolt_anomaly(image_to_process)

        st.subheader("Analysis Result:")
        if "Normal" in predicted_class: # Check if 'Normal' is in the class name
            st.success(f"This product is: **{predicted_class}** (Confidence: {confidence:.2f}%)")
            st.balloons() # Fun little animation for good results
        else:
            st.error(f"Anomaly Detected: **{predicted_class}** (Confidence: {confidence:.2f}%)")
        st.write("---")
    else:
        st.warning("Please upload an image or take a picture to analyze.")

