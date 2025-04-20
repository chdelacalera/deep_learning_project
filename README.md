# **Multi-Model Transfer Learning for Image Classification**

Team members:

| Name                           | Email                               |
| -----------------------        | ----------------------------------- |
| Margarita Vera Cabrer          | 202406918@alu.comillas.edu       |
| Elena Martínez Torrijos        | 202407060@alu.comillas.edu          |
| Claudia Hermández de la Calera | chdelacalera@alu.comillas.edu       |
| Javier Gallergo Fernández      | 201905882@alu.icai.comillas.edu     |


This project focuses on developing and evaluating an image classification system using various deep learning models. The goal is to classify images into 15 different categories, such as different types of rooms (e.g., Bedroom, Kitchen, Office) and urban environments (e.g., Forest, Highway, Mountain), using pre-trained models like ResNet and EfficientNet. The models are fine-tuned to optimize their performance on a specific dataset.

## Project Structure

```
.
├── src/                                     # Source code
|   ├── app_streamlit.py                     # Streamlit App
│   ├── cnn.py                               # CNN model architecture definition
│   └── utils.py                             # Useful load functions
├── models/                                  # Saved model weights
├── wandb/                                   # Weights & Biases data for experiment tracking
├── Multi_Model_Transfer_Learning.ipynb      # Notebook for training and comparing multiple models using transfer learning
└── README.md                                # This file
```

## Models Tested

The following models were trained and tested for this image classification task:

1. **ResNet50**
2. **ResNet50-HighLR**: Variant with a higher learning rate for faster convergence.
3. **ResNet50-LowLR**: Variant with a lower learning rate for finer adjustments.
4. **ResNet50-MoreLayers**: ResNet50 with more layers unfrozen for more detailed feature extraction.
5. **ResNet18**: A smaller variant of ResNet, optimized for quicker training times with lower computational resources.
6. **EfficientNet-B0**: EfficientNet model, known for its efficiency and high performance with fewer parameters.
7. **ResNet101**: A deeper ResNet model offering higher performance, especially in more complex image recognition tasks.
8. **ResNet101-HigherReg**: ResNet101 with higher regularization to avoid overfitting.

Each model was evaluated based on its validation accuracy, training time, and performance across different metrics, including training/validation loss, accuracy, and confusion matrix.


## Selecting the Best Model and Final Evaluation

### Model Selection

After training all the models, the best-performing model is selected based on its **validation accuracy**. The model with the highest validation accuracy is considered the most reliable and is saved for future use.

- The best model in this project was **ResNet101**, with a **validation accuracy of 94.07%**.
- This model showed the best balance between accuracy and generalization across the validation dataset.

The selected model was saved under the name:
```bash
best_model_resnet101.pt
