import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaClassifier, self).__init__()
        # CNN architecture for chest X-ray classification
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images and labels
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpeg'):
                    self.images.append(str(img_path))
                    self.labels.append(0 if class_name == 'NORMAL' else 1)
        
        print(f"Loaded {len(self.images)} images")
        print(f"Normal: {self.labels.count(0)}, Pneumonia: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def download_dataset(data_dir="chest_xray_data"):
    """Download and extract the chest X-ray dataset"""
    print("Note: This script expects the Kaggle chest X-ray dataset to be available.")
    print("Please download the dataset from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")
    print("Extract it to a folder named 'chest_xray' in the current directory.")
    print("The directory structure should be:")
    print("chest_xray/")
    print("  train/")
    print("    NORMAL/")
    print("    PNEUMONIA/")
    print("  test/")
    print("    NORMAL/")
    print("    PNEUMONIA/")
    print("  val/")
    print("    NORMAL/")
    print("    PNEUMONIA/")
    
    if not os.path.exists("chest_xray"):
        raise FileNotFoundError("Please download and extract the Kaggle chest X-ray dataset as described above.")
    
    return "chest_xray"

def get_data_transforms():
    """Define data transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])  # Grayscale normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    
    return train_transform, val_transform

def prepare_data(data_dir, batch_size=32):
    """Load and prepare the chest X-ray dataset"""
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        os.path.join(data_dir, 'train'), 
        transform=train_transform
    )
    val_dataset = ChestXrayDataset(
        os.path.join(data_dir, 'val'), 
        transform=val_transform
    )
    test_dataset = ChestXrayDataset(
        os.path.join(data_dir, 'test'), 
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cpu'):
    """Train the model and return training history"""
    model.to(device)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print('-' * 60)
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model on test data"""
    model.to(device)
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, desc='Testing'):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    return accuracy, all_predictions, all_targets

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def convert_to_onnx(model, input_size=(1, 1, 224, 224), onnx_path="pneumonia_model.onnx", device='cpu'):
    """Convert PyTorch model to ONNX format"""
    model.to(device)
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(input_size).to(device)
    
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

def verify_onnx_model(onnx_path, test_loader, device='cpu'):
    """Verify ONNX model works correctly"""
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Test with a small batch
    for batch_X, batch_y in test_loader:
        test_input = batch_X[:2].numpy()  # Take first 2 samples
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"ONNX model verification successful. Output shape: {ort_outputs[0].shape}")
        break
    
    return True

def main():
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set MLflow experiment
    mlflow.set_experiment("pneumonia_classification")
    
    with mlflow.start_run() as run:
        print("Starting MLflow experiment...")
        
        # Hyperparameters
        learning_rate = 0.001
        num_epochs = 25
        batch_size = 32
        
        # Log hyperparameters
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("device", str(device))
        
        # Download and prepare data
        print("Preparing data...")
        try:
            data_dir = download_dataset()
            train_loader, val_loader, test_loader = prepare_data(data_dir, batch_size)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        
        # Initialize model, loss function, and optimizer
        model = PneumoniaClassifier(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("trainable_parameters", trainable_params)
        
        print(f"Model has {total_params:,} total parameters")
        print(f"Model has {trainable_params:,} trainable parameters")
        
        print("Training model...")
        # Train the model
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, criterion, optimizer, num_epochs, device
        )
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pth'))
        
        # Evaluate the model
        print("Evaluating model...")
        test_accuracy, predictions, targets = evaluate_model(model, test_loader, device)
        
        # Log metrics
        mlflow.log_metric("final_train_accuracy", train_accuracies[-1])
        mlflow.log_metric("final_val_accuracy", val_accuracies[-1])
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("final_train_loss", train_losses[-1])
        mlflow.log_metric("final_val_loss", val_losses[-1])
        
        # Log training curves
        for epoch, (t_loss, t_acc, v_loss, v_acc) in enumerate(zip(train_losses, train_accuracies, val_losses, val_accuracies)):
            mlflow.log_metric("train_loss", t_loss, step=epoch)
            mlflow.log_metric("train_accuracy", t_acc, step=epoch)
            mlflow.log_metric("val_loss", v_loss, step=epoch)
            mlflow.log_metric("val_accuracy", v_acc, step=epoch)
        
        # Create and log classification report
        class_names = ['NORMAL', 'PNEUMONIA']
        report = classification_report(targets, predictions, target_names=class_names)
        print("Classification Report:")
        print(report)
        
        # Save classification report as artifact
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        
        # Create and log confusion matrix
        plot_confusion_matrix(targets, predictions, class_names)
        mlflow.log_artifact("confusion_matrix.png")
        
        # Save model state dict as artifact
        mlflow.log_artifact("best_model.pth")
        
        # Log the PyTorch model to MLflow
        print("Logging PyTorch model to MLflow...")
        mlflow.pytorch.log_model(
            model, 
            "pneumonia_classifier_pytorch",
            registered_model_name="pneumonia_pytorch_classifier"
        )
        
        # Convert model to ONNX
        print("Converting model to ONNX...")
        onnx_path = "pneumonia_model.onnx"
        convert_to_onnx(model, input_size=(1, 1, 224, 224), onnx_path=onnx_path, device=device)
        
        # Verify ONNX model
        verify_onnx_model(onnx_path, test_loader, device)
        
        # Log the ONNX model to MLflow
        print("Logging ONNX model to MLflow...")
        onnx_model = onnx.load(onnx_path)
        mlflow.onnx.log_model(
            onnx_model,
            "pneumonia_classifier_onnx",
            registered_model_name="pneumonia_onnx_classifier"
        )
        
        print(f"Models logged successfully!")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Clean up temporary files
        files_to_clean = [
            "classification_report.txt", 
            "best_model.pth", 
            "confusion_matrix.png",
            onnx_path
        ]
        for file_path in files_to_clean:
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    main() 