import streamlit as st
import os
import torch
from torchvision import transforms
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import glob
import re
from cnn import CNN
from utils import get_pretrained_model, predict_sample_images

# Function to get model display name from filename
def get_model_display_name(filename):
    """Extract a readable model name from filename."""
    # Extract base name (remove path and extension)
    base_name = os.path.basename(filename).replace('.pt', '')
    
    # Try to match naming patterns
    if 'best_acc_' in base_name:
        base_name = base_name.replace('best_acc_', 'Best Accuracy: ')
    elif 'best_model_' in base_name:
        base_name = base_name.replace('best_model_', 'Best Model: ')
    elif '_finetune' in base_name:
        base_name = base_name.replace('_finetune', ' Fine-tuned')
    
    # Format the name more nicely
    base_name = base_name.replace('_', ' ').title()
    
    return base_name

# Function to get model architecture from filename
def get_model_architecture(filename):
    """Extract model architecture from filename."""
    if 'resnet18' in filename.lower():
        return 'resnet18'
    elif 'resnet50' in filename.lower():
        return 'resnet50'
    elif 'resnet101' in filename.lower():
        return 'resnet101'
    elif 'efficientnet_b0' in filename.lower():
        return 'efficientnet_b0'
    elif 'mobilenet_v3' in filename.lower():
        return 'mobilenet_v3_large'
    elif 'vgg16' in filename.lower():
        return 'vgg16'
    else:
        return 'resnet50'  # Default to resnet50 if can't determine

# Function to preprocess image before prediction
def load_image(image_file):
    """
    Load and preprocess an image from a file.
    
    Args:
        image_file: File object containing the image
        
    Returns:
        PIL Image object in RGB format
    """
    image = Image.open(image_file)
    image = image.convert("RGB")
    return image

# Function to predict a single image
def predict_single_image(model, image, transform, device):
    """
    Make a prediction for a single image.
    
    Args:
        model: Trained CNN model
        image: PIL Image to classify
        transform: Transforms to apply to the image
        device: Device to run inference on
        
    Returns:
        Predicted class index and class probabilities
    """
    model.eval()
    with torch.no_grad():
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, 1)
    
    return predicted_class.item(), probabilities[0].cpu().numpy()

# Streamlit application
def main():
    """
    Main function for the Streamlit application.
    Sets up the UI and handles the image classification workflow.
    """
    st.title("Scene Classification with Transfer Learning")
    st.write("This app uses transfer learning with various pre-trained models to classify scene images.")
    
    # Class names
    class_names = [
        "Bedroom", "Coast", "Forest", "Highway", "Industrial", 
        "Inside city", "Kitchen", "Living room", "Mountain", "Office", "Open country", 
        "Store", "Street", "Suburb", "Tall building"
    ]
    
    # Set up sidebar
    st.sidebar.title("Model Selection")
    
    # Path to the models directory
    models_dir = "models"
    
    # Get available model files
    model_files = []
    
    # Check if models directory exists
    if os.path.exists(models_dir):
        # Find all .pt files in the models directory
        model_files = glob.glob(os.path.join(models_dir, "*.pt"))
        # Sort them by name
        model_files.sort()
    
    if not model_files:
        st.error("No trained models found in the 'models' directory!")
        st.info("Please train some models or check the directory path.")
        return
    
    # Create a dictionary mapping display names to file paths
    model_options = {get_model_display_name(f): f for f in model_files}
    
    # Find the best model (one that starts with "best_model_") to use as default
    default_index = 0
    best_model_key = None
    
    for i, (key, path) in enumerate(model_options.items()):
        if "best_model_" in os.path.basename(path).lower():
            default_index = i
            best_model_key = key
            break
    
    # Create a dropdown selector for the model
    selected_model_display = st.sidebar.selectbox(
        "Select a trained model",
        options=list(model_options.keys()),
        index=default_index  # Set the default to the best model
    )
    
    # Get the selected model file path
    selected_model_path = model_options[selected_model_display]
    
    # Determine the model architecture
    model_architecture = get_model_architecture(selected_model_path)
    
    # Display model information in sidebar
    st.sidebar.subheader("Model Info")
    st.sidebar.write(f"Architecture: {model_architecture}")
    st.sidebar.write(f"Model File: {os.path.basename(selected_model_path)}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.write(f"Using device: {device}")
    
    # Load the pre-trained base model
    try:
        base_model = get_pretrained_model(model_architecture)
        model = CNN(base_model, len(class_names), unfreezed_layers=0)
        
        # Load the trained weights
        model.load_state_dict(torch.load(selected_model_path, map_location=device))
        model = model.to(device)
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Create the transforms for inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Image upload section
    st.subheader("Upload an Image")
    
    # Two options: file upload or camera input
    option = st.radio("Input Method", ["Upload Image", "Use Camera"])
    
    image = None
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Select an image to classify", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            st.success("Image uploaded successfully!")
    else:  # Use Camera
        camera_image = st.camera_input("Take a photo")
        if camera_image is not None:
            image = load_image(camera_image)
            st.success("Photo captured successfully!")
    
    # If an image was provided, display and classify it
    prediction_made = False
    predicted_class = None
    probabilities = None
    
    if image is not None:
        # Display the image
        st.subheader("Image")
        st.image(image, caption="Input Image", use_column_width=True)
        
        # Prediction button
        if st.button("Classify Image"):
            with st.spinner("Processing..."):
                # Get prediction
                predicted_class, probabilities = predict_single_image(model, image, transform, device)
                prediction_made = True
                
                # Display result
                st.subheader("Classification Result")
                st.success(f"Predicted Class: **{class_names[predicted_class]}**")
                
                # Display top probabilities
                top_k = 5  # Show top 5 classes
                top_probs, top_classes = torch.tensor(probabilities).topk(min(top_k, len(class_names)))
                
                # Create a bar chart
                fig, ax = plt.subplots(figsize=(10, 4))
                y_pos = range(min(top_k, len(class_names)))
                
                # Get class names for top probabilities
                class_labels = [class_names[i] for i in top_classes]
                
                # Create bar chart
                bars = ax.barh(y_pos, top_probs, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(class_labels)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel('Probability')
                ax.set_title('Top Predictions')
                
                # Add probability values as text
                for i, bar in enumerate(bars):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{top_probs[i]:.4f}', va='center')
                
                # Display the chart
                st.pyplot(fig)
    
    # Add information about the dataset
    with st.sidebar.expander("Dataset Information"):
        st.write("**Number of Classes:** 15")
        st.write("**Classes:**")
        st.write(", ".join(class_names))
        
    # Add model comparison option - only show checkbox if prediction was made
    if image is not None:
        show_comparison = st.sidebar.checkbox("Compare with another model", 
                                             disabled=not prediction_made,
                                             help="First make a prediction with the current model" if not prediction_made else None)
        
        if show_comparison and prediction_made:
            st.sidebar.subheader("Comparison Model")
            
            # Remove the currently selected model from options
            comparison_options = {k: v for k, v in model_options.items() if k != selected_model_display}
            
            if comparison_options:
                comparison_model_display = st.sidebar.selectbox(
                    "Select model to compare with",
                    options=list(comparison_options.keys())
                )
                
                comparison_model_path = comparison_options[comparison_model_display]
                comparison_model_architecture = get_model_architecture(comparison_model_path)
                
                if st.sidebar.button("Run Comparison"):
                    with st.spinner("Running comparison..."):
                        # Load comparison model
                        comparison_base_model = get_pretrained_model(comparison_model_architecture)
                        comparison_model = CNN(comparison_base_model, len(class_names), unfreezed_layers=0)
                        comparison_model.load_state_dict(torch.load(comparison_model_path, map_location=device))
                        comparison_model = comparison_model.to(device)
                        
                        # Get prediction from comparison model
                        comp_class, comp_probs = predict_single_image(comparison_model, image, transform, device)
                        
                        # Display comparison results
                        st.subheader(f"Comparison: {comparison_model_display}")
                        st.success(f"Predicted Class: **{class_names[comp_class]}**")
                        
                        # Create comparison table
                        st.subheader("Model Comparison")
                        data = {
                            "Model": [selected_model_display, comparison_model_display],
                            "Predicted Class": [class_names[predicted_class], class_names[comp_class]],
                            f"Confidence in '{class_names[predicted_class]}'": [
                                f"{probabilities[predicted_class]:.4f}",
                                f"{comp_probs[predicted_class]:.4f}"
                            ],
                            f"Confidence in '{class_names[comp_class]}'": [
                                f"{probabilities[comp_class]:.4f}",
                                f"{comp_probs[comp_class]:.4f}"
                            ]
                        }
                        
                        st.table(data)
            else:
                st.sidebar.info("No other models available for comparison.")

if __name__ == "__main__":
    main()
