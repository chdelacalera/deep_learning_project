import streamlit as st
import os
import torch
from torchvision import transforms
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from torch import nn

# Definir el modelo CNN tal como lo tienes (sin la parte de entrenamiento)
class CNN(nn.Module):
    """Convolutional Neural Network model for image classification."""
    
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Reemplazar la capa final del modelo base
        self.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

        self.base_model.fc = nn.Identity()

    def forward(self, x):
        """Forward pass of the model."""
        x = self.base_model(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def predict(self, image, transform):
        """Predicción para una imagen"""
        self.eval()
        image = transform(image).unsqueeze(0)  # Transforma la imagen y agrega dimensión de batch
        output = self(image)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()

    def load_model(self, filename: str):
        """Cargar el modelo guardado"""
        # Cargar el estado del modelo desde el archivo
        self.load_state_dict(torch.load(filename))
        self.eval()

# Función para preprocesar la imagen antes de la predicción
def load_image(image_file):
    image = Image.open(image_file)
    image = image.convert("RGB")
    return image

# Aplicación Streamlit
def main():
    st.title("Aplicación de Clasificación de Imágenes con CNN")

    # Ruta del modelo entrenado
    model_filename = '/Users/elenamartineztorrijos/Desktop/MBD/Parte2/ML2/deep/models/resnet50-1epoch.pt'
    model = CNN(torchvision.models.resnet50(pretrained=True), num_classes=15)  # Ajusta num_classes si es necesario
    
    if os.path.exists(model_filename):
        model.load_model(model_filename)
        st.success(f"¡Modelo '{model_filename}' cargado correctamente!")

    # Cargar imagen desde el archivo
    uploaded_file = st.file_uploader("Selecciona una imagen para clasificar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Cargar la imagen seleccionada por el usuario
        image = load_image(uploaded_file)

        # Mostrar la imagen cargada
        st.header("Imagen Cargada")
        st.image(image, caption="Imagen cargada para predicción", use_column_width=True)

        # Transformación de la imagen
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        
        # Realizar la predicción
        prediction = model.predict(image, transform)
        
        # Mostrar el resultado
        class_names = [
            "Bedroom", "Coast", "Forest", "Highway", "Industrial", 
            "Inside city", "Kitchen", "Living room", "Mountain", "Office", "Open country", 
            "Store", "Street", "Suburb", "Tall building"
        ]
        st.write(f"Predicción: {class_names[prediction]}")

if __name__ == "__main__":
    main()
