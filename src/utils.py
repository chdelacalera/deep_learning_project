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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
import wandb
from src.cnn import *


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch.
    
    Args:
        model: Model to train.
        train_loader: DataLoader with training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on (cuda or cpu).
        
    Returns:
        train_loss: Average training loss.
        train_acc: Training accuracy.
    """
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = correct / total
    
    return train_loss, train_acc


def validate(model, val_loader, criterion, device):
    """Validate the model.
    
    Args:
        model: Model to validate.
        val_loader: DataLoader with validation data.
        criterion: Loss function.
        device: Device to validate on (cuda or cpu).
        
    Returns:
        val_loss: Average validation loss.
        val_acc: Validation accuracy.
        all_preds: List of all predictions.
        all_labels: List of all true labels.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / total
    val_acc = correct / total
    
    return val_loss, val_acc, all_preds, all_labels


def train_model_with_early_stopping(model, train_loader, valid_loader, criterion, optimizer, 
                                  device, num_epochs=20, patience=5):
    """Train the model with early stopping.
    
    Args:
        model: Model to train.
        train_loader: DataLoader with training data.
        valid_loader: DataLoader with validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on (cuda or cpu).
        num_epochs: Maximum number of epochs to train.
        patience: Number of epochs to wait for improvement before stopping.
        
    Returns:
        model: Trained model.
        history: Dictionary containing training and validation metrics.
    """
    # Initialize variables for early stopping
    best_val_acc = 0
    epochs_no_improve = 0
    best_model_path = "models/best_model_temp.pt"
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Initialize history dictionary
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'epoch': []
    }
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, valid_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch'].append(epoch + 1)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    
    # Remove temporary model file
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        
    return model, history

def train_single_model(model_name, model_config, common_config, num_classes, device, train_loader, valid_loader, class_names):
    """Train a single model with the specified configuration and log to W&B."""
    print(f"\n{'-'*50}")
    print(f"Training {model_name}")
    print(f"{'-'*50}\n")
    
    # 1) Start run of W&B
    run = wandb.init(
        project="deep-lab-models",
        name=f"run-{model_name}",
        group="multi-model-transfer-learning",
        config={**common_config, **model_config}
    )

    # 2) Obtain and prepare the model
    base_model = get_pretrained_model(model_config["base_model"])
    print(f"Using {model_config['base_model']} as base model")
    model = CNN(base_model, num_classes, model_config["unfreeze_layers"]).to(device)

    # Show trainable parameters
    print("\nTrainable layers:")
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
            trainable_params += param.numel()
    print(f"\nTotal trainable parameters: {trainable_params:,}")

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"]
    )

    # 3) Training with stopping and logging
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(common_config["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, valid_loader, criterion, device)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Manual early stopping 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            torch.save(model.state_dict(), f"best_{model_name}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= common_config["patience"]:
                print("Early stopping triggered.")
                break

    # 4) Log 
    fig_cm = plot_confusion_matrix(val_labels, val_preds, class_names)
    wandb.log({"confusion_matrix": wandb.Image(fig_cm)})

    sample_fig = predict_sample_images(model, valid_loader, device, class_names, num_samples=5)
    wandb.log({"sample_predictions": wandb.Image(sample_fig)})

    # 5) Save final model
    model_filename = f"{model_name.lower().replace('-', '_')}_finetune.pt"
    torch.save(model.state_dict(), model_filename)
    artifact = wandb.Artifact(f"{model_name}-model", type="model")
    artifact.add_file(model_filename)
    run.log_artifact(artifact)

    run.finish()

    val_acc = history["val_acc"][-1]
    training_time = None  
    return {
        "model": model,
        "history": history,
        "val_preds": val_preds,
        "val_labels": val_labels,
        "val_acc": val_acc,
        "training_time": training_time,
        "config": {**model_config, **common_config}
    }



def plot_training_history(history):
    """Plot the training and validation loss and accuracy.
    
    Args:
        history: Dictionary containing training and validation metrics.
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    ax1.plot(history['epoch'], history['train_loss'], label='Training Loss')
    ax1.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot training and validation accuracy
    ax2.plot(history['epoch'], history['train_acc'], label='Training Accuracy')
    ax2.plot(history['epoch'], history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(true_labels, predictions, class_names):
    """Plot confusion matrix.
    
    Args:
        true_labels: True labels.
        predictions: Predicted labels.
        class_names: List of class names.
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.show()


def predict_sample_images(model, data_loader, device, class_names, num_samples=4):
    """Predict and display sample images from the dataset.
    
    Args:
        model: Trained model.
        data_loader: DataLoader with data to sample from.
        device: Device to run prediction on.
        class_names: List of class names.
        num_samples: Number of samples to display.
    """
    model.eval()
    
    # Get sample indices
    dataset = data_loader.dataset
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    for i, idx in enumerate(indices):
        # Get image and label
        img, label = dataset[idx]
        
        # Make prediction
        img_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            _, pred = torch.max(output, 1)
        
        # Convert tensor image to numpy for display
        img_np = img.cpu().numpy().transpose((1, 2, 0))
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Display image
        axes[i].imshow(img_np)
        color = 'green' if pred.item() == label else 'red'
        axes[i].set_title(f"True: {class_names[label]}\nPred: {class_names[pred.item()]}", color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def get_pretrained_model(model_name):
    """Get a pre-trained model by name."""
    if model_name == 'resnet18':
        return torchvision.models.resnet18(weights='DEFAULT')
    elif model_name == 'resnet34':
        return torchvision.models.resnet34(weights='DEFAULT')
    elif model_name == 'resnet50':
        return torchvision.models.resnet50(weights='DEFAULT')
    elif model_name == 'resnet101':
        return torchvision.models.resnet101(weights='DEFAULT')
    elif model_name == 'resnet152':
        return torchvision.models.resnet152(weights='DEFAULT')
    elif model_name == 'efficientnet_b0':
        return torchvision.models.efficientnet_b0(weights='DEFAULT')
    elif model_name == 'efficientnet_b1':
        return torchvision.models.efficientnet_b1(weights='DEFAULT')
    elif model_name == 'efficientnet_b2':
        return torchvision.models.efficientnet_b2(weights='DEFAULT')
    elif model_name == 'mobilenet_v3_small':
        return torchvision.models.mobilenet_v3_small(weights='DEFAULT')
    elif model_name == 'mobilenet_v3_large':
        return torchvision.models.mobilenet_v3_large(weights='DEFAULT')
    elif model_name == 'vgg16':
        return torchvision.models.vgg16(weights='DEFAULT')
    elif model_name == 'vgg19':
        return torchvision.models.vgg19(weights='DEFAULT')
    elif model_name == 'densenet121':
        return torchvision.models.densenet121(weights='DEFAULT')
    elif model_name == 'inception_v3':
        return torchvision.models.inception_v3(weights='DEFAULT')
    else:
        raise ValueError(f"Unsupported model: {model_name}") 