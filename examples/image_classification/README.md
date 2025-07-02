# Pneumonia Classification with MLflow and ONNX

This directory contains a complete implementation of a pneumonia classification model using chest X-ray images. The model is built with PyTorch, tracked with MLflow, and exported to ONNX format for deployment.

## Overview

The `pneumonia_classification_mlflow_onnx.py` script implements:
- A CNN model for binary classification of chest X-rays (Normal vs Pneumonia)
- Complete MLflow experiment tracking
- Model export to ONNX format for deployment
- Comprehensive evaluation with confusion matrix and classification reports

## Dataset

This model uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:
- **Source**: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
- **Size**: ~5,856 chest X-ray images (JPEG)
- **Classes**: Normal and Pneumonia
- **Training**: 5,216 images
- **Validation**: 16 images  
- **Test**: 624 images

### Dataset Setup

1. **Download the dataset**:
   - Go to https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
   - Download the dataset zip file
   - Extract it to a folder named `chest_xray` in your working directory

2. **Expected directory structure**:
   ```
   chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   │   └── *.jpeg files
   │   └── PNEUMONIA/
   │       └── *.jpeg files
   ├── val/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── test/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start MLflow server** (optional, for UI):
   ```bash
   mlflow server --host 127.0.0.1 --port 8080
   ```

## Usage

### Basic Training

Run the complete training pipeline:

```bash
python pneumonia_classification_mlflow_onnx.py
```

This will:
- Load and preprocess the chest X-ray dataset
- Train a CNN model for 25 epochs
- Track all metrics and parameters in MLflow
- Evaluate on test set
- Export model to ONNX format
- Register models in MLflow Model Registry

### Model Architecture

The CNN model includes:
- **Input**: Grayscale chest X-ray images (224x224)
- **Feature Extraction**: 4 convolutional blocks with batch normalization
- **Pooling**: Max pooling and global average pooling
- **Classifier**: Fully connected layers with dropout
- **Output**: Binary classification (Normal vs Pneumonia)

**Model Summary**:
- Input channels: 1 (grayscale)
- Convolutional layers: 32 → 64 → 128 → 256 filters
- Final layers: 256 → 128 → 2 classes
- Total parameters: ~1.2M

### Data Preprocessing

- **Resize**: All images resized to 224x224
- **Normalization**: Pixel values normalized to [0,1] range
- **Augmentation** (training only):
  - Random rotation (±10 degrees)
  - Random horizontal flip
- **Format**: Converted to grayscale tensors

### MLflow Tracking

The script tracks:
- **Parameters**: Learning rate, batch size, epochs, model architecture
- **Metrics**: Training/validation loss and accuracy per epoch
- **Artifacts**: Model weights, classification reports, confusion matrix
- **Models**: Both PyTorch and ONNX versions

Access MLflow UI at: http://127.0.0.1:8080

### Model Export

Two model formats are saved:
1. **PyTorch Model**: Native PyTorch format for Python deployment
2. **ONNX Model**: Cross-platform format for production deployment

## Configuration

Key hyperparameters (modify in `main()` function):

```python
learning_rate = 0.001    # Adam optimizer learning rate
num_epochs = 25          # Training epochs
batch_size = 32          # Batch size for training
```

## Results

Expected performance on the chest X-ray dataset:
- **Training Accuracy**: ~95-98%
- **Validation Accuracy**: ~85-90%
- **Test Accuracy**: ~85-90%

The model generates:
- Classification report with precision, recall, F1-score
- Confusion matrix visualization
- Training curves (loss and accuracy)

## GPU Support

The script automatically detects and uses CUDA if available:
- **CPU**: Works with CPU-only PyTorch
- **GPU**: Automatically utilizes CUDA for faster training

## GUI Applications

### Production Inference GUI

**File:** `pneumonia_inference_gui.py`

A complete Gradio web interface for pneumonia classification inference using the Open Inference protocol:

```bash
python pneumonia_inference_gui.py
```

**Features:**
- Web-based image upload interface
- Real-time inference against deployed models
- Open Inference protocol compatibility
- Detailed classification results with confidence scores
- Connection testing and model metadata display
- Professional medical-grade UI design

**Requirements:**
- Deployed ONNX model on inference server (Triton, etc.)
- Valid authentication token
- Network access to inference endpoint

### Demo GUI (No Deployment Required)

**File:** `demo_inference_gui.py`

A demonstration version with simulated inference for testing the interface:

```bash
python demo_inference_gui.py
```

**Features:**
- Same interface as production version
- Mock inference client with realistic predictions
- No external dependencies or deployments needed
- Perfect for testing and development
- Educational demonstration of the workflow

**Usage:**
- Upload any chest X-ray image
- Get simulated classification results
- Test the interface before production deployment

### GUI Usage Instructions

1. **Start the application:**
   ```bash
   # For production (requires deployed model)
   python pneumonia_inference_gui.py
   
   # For demo (no deployment needed)
   python demo_inference_gui.py
   ```

2. **Access the interface:**
   - Open browser to `http://localhost:7860`
   - Interface will be available immediately

3. **Configure connection:**
   - Enter your model endpoint URL
   - Specify the deployed model name
   - Test connection to verify setup

4. **Upload and analyze:**
   - Upload chest X-ray image (JPEG/PNG)
   - Click "Analyze X-Ray"
   - Review classification results and confidence scores

## File Structure

```
examples/image_classification/
├── pneumonia_classification_mlflow_onnx.py  # Main training script
├── pneumonia_inference_gui.py              # Production GUI app
├── demo_inference_gui.py                   # Demo GUI app
├── requirements.txt                        # Python dependencies
└── README.md                              # This file
```

## Model Registry

Models are automatically registered in MLflow:
- **PyTorch Model**: `pneumonia_pytorch_classifier`
- **ONNX Model**: `pneumonia_onnx_classifier`

## Production Deployment

The ONNX model can be deployed using:
- **ONNX Runtime**: For Python/C++ applications
- **Azure ML**: Direct ONNX deployment
- **TensorFlow.js**: Convert for web deployment
- **Mobile**: ONNX Runtime Mobile for iOS/Android

## Troubleshooting

**Common Issues**:

1. **Dataset not found**: Ensure the `chest_xray` folder is in your working directory
2. **Out of memory**: Reduce batch size if using GPU with limited memory
3. **Import errors**: Install all requirements with `pip install -r requirements.txt`
4. **MLflow UI not accessible**: Start MLflow server with `mlflow server`

## License

This implementation is based on publicly available medical imaging research and follows ethical AI guidelines for healthcare applications.

## References

- Dataset: Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Chest X-Ray Images (Pneumonia)", Mendeley Data, V1
- Original Paper: Kermany et al. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning" (Cell, 2018) 