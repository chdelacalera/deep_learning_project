import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tempfile import TemporaryDirectory
import json

class CNN(nn.Module):
    """Convolutional Neural Network model for image classification."""
    
    def __init__(self, base_model, num_classes, unfreezed_layers=0):
        """CNN model initializer.

        Args:
            base_model: Pre-trained model to use as the base.
            num_classes: Number of classes in the dataset.
            unfreezed_layers: Number of layers to unfreeze from the base model.

        """
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Freeze convolutional layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze specified number of layers
        if unfreezed_layers > 0:
            for layer in list(self.base_model.children())[-unfreezed_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Determine the input features based on model architecture
        if hasattr(self.base_model, 'fc'):
            # ResNet, DenseNet, etc.
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif hasattr(self.base_model, 'classifier'):
            # MobileNetV3, EfficientNet
            if isinstance(self.base_model.classifier, nn.Sequential):
                # For EfficientNet
                in_features = self.base_model.classifier[1].in_features
            else:
                # For MobileNetV3
                in_features = self.base_model.classifier[-1].in_features
            self.base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported base model architecture: {type(self.base_model).__name__}")

        # Add a new softmax output layer
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input data.
        """
        x = self.base_model(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_model(self, 
                    train_loader, 
                    valid_loader, 
                    optimizer, 
                    criterion, 
                    epochs, 
                    nepochs_to_save=10):
        """Train the model and save the best one based on validation accuracy.
        
        Args:
            train_loader: DataLoader with training data.
            valid_loader: DataLoader with validation data.
            optimizer: Optimizer to use during training.
            criterion: Loss function to use during training.
            epochs: Number of epochs to train the model.
            nepochs_to_save: Number of epochs to wait before saving the model.

        Returns:
            history: A dictionary with the training history.
        """
        with TemporaryDirectory() as temp_dir:
            best_model_path = os.path.join(temp_dir, 'best_model.pt')
            best_accuracy = 0.0
            torch.save(self.state_dict(), best_model_path)

            history = {'train_loss': [], 'train_accuracy': [], 'valid_loss': [], 'valid_accuracy': []}
            for epoch in range(epochs):
                self.train()
                train_loss = 0.0
                train_accuracy = 0.0
                for images, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_accuracy += (outputs.argmax(1) == labels).sum().item()

                train_loss /= len(train_loader)
                train_accuracy /= len(train_loader.dataset)
                history['train_loss'].append(train_loss)
                history['train_accuracy'].append(train_accuracy)

                print(f'Epoch {epoch + 1}/{epochs} - '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Train Accuracy: {train_accuracy:.4f}')
                
                
                self.eval()
                valid_loss = 0.0
                valid_accuracy = 0.0
                for images, labels in valid_loader:
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    valid_accuracy += (outputs.argmax(1) == labels).sum().item()

                valid_loss /= len(valid_loader)
                valid_accuracy /= len(valid_loader.dataset)
                history['valid_loss'].append(valid_loss)
                history['valid_accuracy'].append(valid_accuracy)

                print(f'Epoch {epoch + 1}/{epochs} - '
                        f'Validation Loss: {valid_loss:.4f}, '
                        f'Validation Accuracy: {valid_accuracy:.4f}')
                
                if epoch % nepochs_to_save == 0:
                    if valid_accuracy > best_accuracy:
                        best_accuracy = valid_accuracy
                        torch.save(self.state_dict(), best_model_path)
                
            torch.save(self.state_dict(), best_model_path)    
            self.load_state_dict(torch.load(best_model_path))
            return history
        
    def predict(self, data_loader):
        """Predict the classes of the images in the data loader.

        Args:
            data_loader: DataLoader with the images to predict.

        Returns:
            predicted_labels: Predicted classes.
        """
        self.eval()
        predicted_labels = []
        for images, _ in data_loader:
            outputs = self(images)
            predicted_labels.extend(outputs.argmax(1).tolist())
        return predicted_labels
        
    def save_model(self, filename: str):
        """Save the model to disk.

        Args:
            filename: Name of the file to save the model.
        """
        # If the directory does not exist, create it
        os.makedirs(os.path.dirname('models/'), exist_ok=True)

        # Full path to the model
        filename = os.path.join('models', filename)

        # Save the model
        torch.save(self.state_dict(), filename+'.pt')

    @staticmethod
    def _plot_training(history):
        """Plot the training history.

        Args:
            history: A dictionary with the training history.
        """
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['valid_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['valid_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

    

        

def load_data(train_dir, valid_dir, batch_size, img_size):
    """Load and transform the training and validation datasets.

    Args:
        train_dir: Path to the training dataset.
        valid_dir: Path to the validation dataset.
        batch_size: Number of images per batch.
        img_size: Expected size of the images.

    Returns:
        train_loader: DataLoader with the training dataset.
        valid_loader: DataLoader with the validation dataset.
        num_classes: Number of classes in the dataset.
    """
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30), # Rotate the image by a random angle
        transforms.RandomResizedCrop(img_size), # Crop the image to a random size and aspect ratio
        transforms.RandomHorizontalFlip(), # Horizontally flip the image with a 50% probability
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, len(train_data.classes)

def load_model_weights(filename: str):
        """Load a model from disk.
        IMPORTANT: The model must be initialized before loading the weights.
        Args:
            filename: Name of the file to load the model.
        """
        # Full path to the model
        filename = os.path.join('models', filename)

        # Load the model
        state_dict = torch.load(filename+'.pt')
        return state_dict