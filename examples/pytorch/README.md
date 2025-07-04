# PyTorch Iris Classification Example

This example demonstrates a complete ML pipeline using PyTorch to train an iris flower classifier and deploy it to Cloudera AI Inference.

## Files Overview

### `iris_classification_notebook.ipynb` - Sample Complete Training, Deployment and Inference flow
- **Purpose**: Illustrates how to train a classification model using Pytorch, and prepare it for deployment to Cloudera AI Inference
- **Features**:
  - Data preprocessing with StandardScaler
  - MLflow experiment tracking and model registry
  - Automatic conversion from PyTorch to ONNX format
  - Model validation and performance metrics
  - Registers both PyTorch and ONNX versions to MLflow
  - Sets up CDP CLI in the workbench project so that the CLI can be used to list services from CPP Control Plane, and to generate workload auth token
  - Use AI Registry API to get model details
  - Use AI Inference API to deploy a model endpoint
  - Use Open Inference Protocol SDK to run a batch inference

### `iris_mlflow_pytorch_onnx.py` - Training Pipeline
- **Purpose**: Trains a neural network classifier on the Iris dataset
- **Architecture**: 3-layer neural network (4 → 16 → 16 → 3) with ReLU activation and dropout
- **Features**:
  - Data preprocessing with StandardScaler
  - MLflow experiment tracking and model registry
  - Automatic conversion from PyTorch to ONNX format
  - Model validation and performance metrics
  - Registers both PyTorch and ONNX versions to MLflow

### `infer_iris.py` - Batch Inference Client
- **Purpose**: Runs batch inference against deployed models on Cloudera AI Inference
- **Features**:
  - Dynamic batching optimization based on Triton server configuration
  - Comprehensive error handling and retry logic
  - Performance metrics (throughput, latency, accuracy)
  - Model configuration introspection
  - Classification report generation

## Usage

1. **Train the model**: 
   ```bash
   python iris_mlflow_pytorch_onnx.py
   ```

2. **Deploy the ONNX model** to your Cloudera AI Inference endpoint

3. **Configure inference script**: Update `infer_iris.py` with your:
   - `BASE_URL` - Your inference endpoint URL
   - `MODEL_NAME` - Your deployed model identifier
   - Modify `load_token()` function for your authentication method

4. **Run batch inference**: 
   ```bash
   python infer_iris.py
   ```