import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io


# --- Streamlit Page Configuration and Title (MUST BE FIRST STREAMLIT COMMANDS) ---
st.set_page_config(page_title="InspectorsAlly - Bolt Quality Check", page_icon=":wrench:")

st.title("InspectorsAlly - Bolt Quality Control")
st.caption("Boost Your Quality Control with AI-Powered Bolt Inspection")
st.write("Upload an image of a bolt or use your camera to classify it as Good / Anomaly.")

# IMPORTANT ASSUMPTION:
# This code assumes that './model/keras_model.h5' is actually a PyTorch model's
# state_dict saved with a .h5 extension, and that the model architecture
# (e.g., CustomVGG) is defined in a 'utils/model.py' file.
# If './model/keras_model.h5' is truly a TensorFlow model, this will NOT work without TensorFlow.

# Attempt to import CustomVGG from utils.model as seen in your other files
try:
    from utils.model import CustomVGG
except ImportError:
    st.error("Error: Could not import CustomVGG from utils.model. "
             "Please ensure 'utils' folder exists and 'model.py' defines CustomVGG.")
    # Define a dummy class to prevent further errors if import fails
    class CustomVGG(torch.nn.Module):
        def __init__(self, num_classes=2):
            super(CustomVGG, self).__init__()
            # Dummy layers to make it runnable, replace with actual VGG structure
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(16 * 7 * 7, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, num_classes),
            )
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x


# --- Model Loading and Class Names ---
# Define the device to run the model on (CPU or GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path and labels path remain as per your request
MODEL_FILE_PATH = './model/keras_model.h5' # This is assumed to be a PyTorch state_dict
LABELS_FILE_PATH = './model/labels.txt'

@st.cache_resource # Use st.cache_resource to load the model only once
def load_pytorch_model(model_path, num_classes=5): # num_classes based on your 5 bolt types
    try:
        # Instantiate the PyTorch model architecture
        # Assuming CustomVGG is defined in utils/model.py and takes num_classes
        model = CustomVGG(num_classes=num_classes)
        
        # Load the state dictionary
        # map_location ensures it loads correctly regardless of original training device
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set the model to evaluation mode
        st.success(f"PyTorch model loaded successfully on {device} from {model_path}!")
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}. Make sure it's a PyTorch state_dict compatible with CustomVGG.")
        return None

# Load the model
model = load_pytorch_model(MODEL_FILE_PATH)

# Load the labels from the specified file path
class_names = []
try:
    with open(LABELS_FILE_PATH, 'r') as f:
        # Assuming labels.txt has format "0 Normal_Bolt", "1 Bent_Bolt"
        class_names = [line.strip().split(' ', 1)[1] for line in f]
    st.success(f"Class names loaded successfully from {LABELS_FILE_PATH}!")
except FileNotFoundError:
    st.error(f"Labels file not found at {LABELS_FILE_PATH}. Please ensure it's in the correct directory.")
    # Fallback if labels.txt is missing, use hardcoded names
    class_names = [
        "Normal_Bolt",
        "Bent_Bolt",
        "Rusted_Bolt",
        "Stripped_Head_Bolt",
        "Damaged_Thread_Bolt"
    ]
    st.warning("Using fallback class names. Ensure these match your model's output order.")
except Exception as e:
    st.error(f"Error loading class names from {LABELS_FILE_PATH}: {e}")
    class_names = [
        "Normal_Bolt",
        "Bent_Bolt",
        "Rusted_Bolt",
        "Stripped_Head_Bolt",
        "Damaged_Thread_Bolt"
    ]
    st.warning("Using fallback class names due to error. Ensure these match your model's output order.")


# --- Image Preprocessing Function (PyTorch Transforms) ---
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
    if not class_names:
        return "Class Names Not Loaded", 0.0

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
    
    # Ensure index is within bounds of class_names
    if predicted_class_index < len(class_names):
        predicted_class_name = class_names[predicted_class_index]
    else:
        predicted_class_name = f"Unknown Class (Index: {predicted_class_index})"
        st.warning(f"Model predicted an index ({predicted_class_index}) outside the defined class names.")

    confidence_score = probabilities[0][predicted_class_index].item() * 100 # Convert to percentage

    return predicted_class_name, confidence_score

# --- Streamlit UI for Input Method Selection ---
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
