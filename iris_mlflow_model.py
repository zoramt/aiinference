import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.pytorch
import mlflow.onnx
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class IrisClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, num_classes=3):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def prepare_data():
    """Load and prepare the iris dataset"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    """Train the model and return training history"""
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_losses, train_accuracies

def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    return accuracy, all_predictions, all_targets

def convert_to_onnx(model, input_size=(1, 4), onnx_path="iris_model.onnx"):
    """Convert PyTorch model to ONNX format"""
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(input_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX format: {onnx_path}")
    return onnx_path

def verify_onnx_model(onnx_path, test_data):
    """Verify ONNX model works correctly"""
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Test with a small batch
    test_input = test_data[:5].numpy()  # Take first 5 samples
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"ONNX model verification successful. Output shape: {ort_outputs[0].shape}")
    return True

def main():
    # Set MLflow experiment
    mlflow.set_experiment("iris_classification")
    
    with mlflow.start_run() as run:
        print("Starting MLflow experiment...")
        
        # Hyperparameters
        hidden_size = 16
        learning_rate = 0.01
        num_epochs = 100
        batch_size = 16
        
        # Log hyperparameters
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        
        # Prepare data
        print("Preparing data...")
        X_train, X_test, y_train, y_test, scaler = prepare_data()
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model, loss function, and optimizer
        model = IrisClassifier(input_size=4, hidden_size=hidden_size, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        print("Training model...")
        # Train the model
        train_losses, train_accuracies = train_model(
            model, train_loader, criterion, optimizer, num_epochs
        )
        
        # Evaluate the model
        print("Evaluating model...")
        test_accuracy, predictions, targets = evaluate_model(model, test_loader)
        
        # Log metrics
        mlflow.log_metric("final_train_accuracy", train_accuracies[-1])
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("final_train_loss", train_losses[-1])
        
        # Log training curves
        for epoch, (loss, acc) in enumerate(zip(train_losses, train_accuracies)):
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("train_accuracy", acc, step=epoch)
        
        # Create and log classification report
        iris = load_iris()
        class_names = iris.target_names
        report = classification_report(targets, predictions, target_names=class_names)
        print("Classification Report:")
        print(report)
        
        # Save classification report as artifact
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        
        # Save model state dict as artifact
        torch.save(model.state_dict(), "model_state_dict.pth")
        mlflow.log_artifact("model_state_dict.pth")
        
        # Log the PyTorch model to MLflow
        print("Logging PyTorch model to MLflow...")
        mlflow.pytorch.log_model(
            model, 
            "iris_classifier_pytorch",
            registered_model_name="iris_pytorch_classifier"
        )
        
        # Convert model to ONNX
        print("Converting model to ONNX...")
        onnx_path = "model.onnx"
        convert_to_onnx(model, input_size=(1, 4), onnx_path=onnx_path)
        
        # Verify ONNX model
        verify_onnx_model(onnx_path, X_test)
        
        # Log the ONNX model to MLflow
        print("Logging ONNX model to MLflow...")
        onnx_model = onnx.load(onnx_path)
        mlflow.onnx.log_model(
            onnx_model,
            "iris_classifier_onnx",
            registered_model_name="iris_onnx_classifier"
        )
        
        print(f"Models logged successfully!")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Clean up temporary files
        import os
        files_to_clean = ["classification_report.txt", "model_state_dict.pth", onnx_path]
        for file_path in files_to_clean:
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    main()