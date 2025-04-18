import streamlit as st
import os
import torch
from torchvision import transforms
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from torch import nn

# Define the CNN model for image classification
class CNN(nn.Module):
    """
    Convolutional Neural Network model for image classification.
    Uses a pre-trained base model with a custom fully connected layer for classification.
    """
    
    def __init__(self, base_model, num_classes):
        """
        Initialize the CNN model.
        
        Args:
            base_model: Pre-trained model to use as feature extractor
            num_classes: Number of output classes for classification
        """
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Replace the final layer of the base model with custom classifier
        self.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

        # Set base model's fully connected layer to identity to use our custom classifier
        self.base_model.fc = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor containing the batch of images
            
        Returns:
            Model predictions
        """
        x = self.base_model(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def predict(self, image, transform):
        """
        Make a prediction for a single image.
        
        Args:
            image: PIL Image to classify
            transform: Transforms to apply to the image
            
        Returns:
            Predicted class index
        """
        self.eval()
        image = transform(image).unsqueeze(0)  # Transform image and add batch dimension
        output = self(image)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()

    def load_model(self, filename: str):
        """
        Load a saved model from disk.
        
        Args:
            filename: Path to the saved model file
        """
        # Load model state from file
        self.load_state_dict(torch.load(filename))
        self.eval()

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

# Streamlit application
def main():
    """
    Main function for the Streamlit application.
    Sets up the UI and handles the image classification workflow.
    """
    st.title("Image Classification App using CNN")

    # Path to the trained model
    model_filename = '/Users/elenamartineztorrijos/Desktop/MBD/Parte2/ML2/deep/models/resnet50-1epoch.pt'
    model = CNN(torchvision.models.resnet50(pretrained=True), num_classes=15)
    
    # Load the trained model if it exists
    if os.path.exists(model_filename):
        model.load_model(model_filename)
        st.success(f"Model '{model_filename}' loaded successfully!")

    # File uploader for image selection
    uploaded_file = st.file_uploader("Select an image to classify", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Load the user-selected image
        image = load_image(uploaded_file)

        # Display the loaded image
        st.header("Uploaded Image")
        st.image(image, caption="Image loaded for prediction", use_column_width=True)

        # Transform the image to the required format
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        
        # Make prediction
        prediction = model.predict(image, transform)
        
        # Display the result with class name
        class_names = [
            "Bedroom", "Coast", "Forest", "Highway", "Industrial", 
            "Inside city", "Kitchen", "Living room", "Mountain", "Office", "Open country", 
            "Store", "Street", "Suburb", "Tall building"
        ]
        st.write(f"Prediction: {class_names[prediction]}")

if __name__ == "__main__":
    main()
