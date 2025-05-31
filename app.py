import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import cv2 # Re-enabled for camera input

# --- Streamlit Page Configuration and Title ---
st.set_page_config(page_title="InspectorsAlly - Bolt Quality Check", page_icon=":wrench:")

st.title("InspectorsAlly - Bolt Quality Control")
st.caption("Boost Your Quality Control with AI-Powered Bolt Inspection")
st.write("Upload an image of a bolt or use your camera to classify it as Good / Anomaly.")

# --- Model Loading and Class Names ---
# Define the device to run the model on (CPU or GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming your PyTorch model is named 'leather_model.h5' and is in a 'weights' folder
# This path matches the structure suggested by your original [Student] streamlitApp.py
# You MUST ensure this file exists and is a valid PyTorch model.
MODEL_PATH = "./weights/leather_model.h5"

@st.cache_resource # Use st.cache_resource to load the model only once
def load_pytorch_model(model_path):
    try:
        # Load the PyTorch model
        model = torch.load(model_path, map_location=device)
        model.eval() # Set the model to evaluation mode
        st.success(f"PyTorch model loaded successfully on {device}!")
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

model = load_pytorch_model(MODEL_PATH)

# Define class names directly, as PyTorch models don't always come with a labels.txt
# Adjust these class names to match your actual bolt anomaly classes
# For example, if your model outputs 0 for Normal_Bolt, 1 for Bent_Bolt, etc.
# IMPORTANT: The order of these names must match the order your model was trained on.
class_names = [
    "Normal_Bolt",
    "Bent_Bolt",
    "Rusted_Bolt",
    "Stripped_Head_Bolt",
    "Damaged_Thread_Bolt"
]

# --- Image Preprocessing Function (PyTorch Transforms) ---
# This transform matches common preprocessing for PyTorch models (e.g., VGG-like models)
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to model's expected input size
    transforms.ToTensor(),         # Convert PIL Image to PyTorch Tensor (scales to [0, 1])
    # Optional: If your model was trained with ImageNet normalization, add this:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Prediction Function ---
def predict_bolt_anomaly(image):
    if model is None:
        return "Model Not Loaded", 0.0

    # Apply preprocessing transforms
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension

    # Move tensor to the correct device
    input_tensor = input_tensor.to(device)

    with torch.no_grad(): # Disable gradient calculation for inference
        output = model(input_tensor)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get the highest confidence prediction
    predicted_class_index = torch.argmax(probabilities).item()
    predicted_class_name = class_names[predicted_class_index]
    confidence_score = probabilities[0][predicted_class_index].item() * 100 # Convert to percentage

    return predicted_class_name, confidence_score

# --- Streamlit UI for Input Method Selection ---
input_method = st.radio("Choose Input Method:", ("File Uploader", "Camera Input"))

uploaded_file_img = None
camera_file_img = None

if input_method == "File Uploader":
    uploaded_file_img = st.file_uploader(
        "Upload an image of a bolt",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a bolt for classification."
    )
elif input_method == "Camera Input":
    # Ensure opencv-python is installed for this to work
    camera_file_img = st.camera_input("Take a picture of a bolt")

# This button will trigger the prediction
submit_button = st.button("Analyze Bolt")

# --- Process Input and Display Results ---
if submit_button:
    image_to_process = None
    if uploaded_file_img is not None:
        image_to_process = Image.open(io.BytesIO(uploaded_file_img.read())).convert("RGB")
    elif camera_file_img is not None:
        image_to_process = Image.open(io.BytesIO(camera_file_img.read())).convert("RGB")

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
