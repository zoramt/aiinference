from open_inference.openapi.client import OpenInferenceClient, InferenceRequest
import httpx
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
from typing import Optional, Dict, Any, List


class TritonBatchInference:
    """Class to handle Triton inference with dynamic batching"""
    
    def __init__(self, base_url: str, model_name: str, token: str):
        self.base_url = base_url
        self.model_name = model_name
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        self.httpx_client = httpx.Client(headers=self.headers)
        self.client = OpenInferenceClient(base_url=base_url, httpx_client=self.httpx_client)
        
    def get_triton_model_config(self) -> Optional[Dict[str, Any]]:
        """Get Triton model configuration including dynamic batching settings"""
        config_url = f"{self.base_url}/v2/models/{self.model_name}/config"
        
        try:
            response = self.httpx_client.get(config_url)
            response.raise_for_status()
            config = response.json()
            print("Model Configuration:")
            print(json.dumps(config, indent=2))
            
            # Extract dynamic batching info
            if 'dynamic_batching' in config:
                return config['dynamic_batching']
            else:
                return None
                
        except Exception as e:
            print(f"Error getting model config: {e}")
            return None
    
    def check_server_status(self) -> bool:
        """Check if the server is ready and get model metadata"""
        try:
            # Check server readiness
            self.client.check_server_readiness()
            print("✓ Server is ready")
            
            # Get model metadata
            metadata = self.client.read_model_metadata(self.model_name)
            metadata_dict = json.loads(metadata.json())
            print("Model Metadata:")
            print(json.dumps(metadata_dict, indent=2))
            
            return True
        except Exception as e:
            print(f"Error checking server status: {e}")
            return False
    
    def prepare_iris_data(self) -> tuple:
        """Load and prepare the iris dataset for inference"""
        # Load iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split data (we'll use test set for batch inference)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale the features (assuming model was trained with scaled features)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Dataset prepared: {len(X_test_scaled)} samples for inference")
        print(f"Feature shape: {X_test_scaled.shape}")
        
        return X_test_scaled, y_test, iris.target_names
    
    def create_batches(self, data: np.ndarray, batch_size: int) -> List[np.ndarray]:
        """Create batches from the input data"""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def run_batch_inference(self, data: np.ndarray, batch_size: int) -> List[np.ndarray]:
        """Run batch inference on the data"""
        batches = self.create_batches(data, batch_size)
        all_predictions = []
        
        print(f"Running inference on {len(batches)} batches of size {batch_size}")
        
        for i, batch in enumerate(batches):
            try:
                # Create inference request
                # Note: Adjust input/output names based on your model's specification
                inference_request = InferenceRequest(
                    inputs=[{
                        "name": "input",  # Adjust based on your model's input name
                        "shape": list(batch.shape),
                        "datatype": "FP32",
                        "data": batch.flatten().tolist()
                    }],
                )
                
                start_time = time.time()
                response = self.client.model_infer(self.model_name, request=inference_request)
                inference_time = time.time() - start_time
                
                # Extract predictions from response
                response_dict = json.loads(response.json())
                output_data = response_dict['outputs'][0]['data']
                
                # Reshape output to match batch size and number of classes
                output_array = np.array(output_data).reshape(batch.shape[0], -1)
                predictions = np.argmax(output_array, axis=1)
                all_predictions.extend(predictions)
                
                print(f"Batch {i+1}/{len(batches)} completed in {inference_time:.3f}s")
                
            except Exception as e:
                print(f"Error in batch {i+1}: {e}")
                # Fill with dummy predictions to maintain consistency
                dummy_predictions = [0] * len(batch)
                all_predictions.extend(dummy_predictions)
        
        return np.array(all_predictions)
    
    def evaluate_predictions(self, predictions: np.ndarray, y_true: np.ndarray, 
                           class_names: List[str]) -> Dict[str, Any]:
        """Evaluate the predictions and return metrics"""
        accuracy = accuracy_score(y_true, predictions)
        report = classification_report(y_true, predictions, target_names=class_names)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'total_samples': len(y_true),
            'correct_predictions': np.sum(predictions == y_true)
        }
        
        return results
    
    def run_full_inference_pipeline(self) -> Dict[str, Any]:
        """Run the complete inference pipeline"""
        print("=" * 50)
        print("TRITON BATCH INFERENCE PIPELINE")
        print("=" * 50)
        
        # Check server status
        if not self.check_server_status():
            return {"error": "Server not ready"}
        
        # Get dynamic batch configuration
        dynamic_batch_config = self.get_triton_model_config()
        
        # Determine batch size
        if dynamic_batch_config:
            print("Dynamic Batching Configuration:")
            print(json.dumps(dynamic_batch_config, indent=2))
            
            # Use preferred batch size or max batch size
            batch_size = dynamic_batch_config.get('preferred_batch_size', [8])[0]
            max_batch_size = dynamic_batch_config.get('max_queue_delay_microseconds', 1000)
            
            print(f"Using batch size: {batch_size}")
        else:
            print("No dynamic batching configuration found, using default batch size: 8")
            batch_size = 8
        
        # Prepare data
        X_test, y_test, class_names = self.prepare_iris_data()
        
        # Run batch inference
        print("\nRunning batch inference...")
        start_time = time.time()
        predictions = self.run_batch_inference(X_test, batch_size)
        total_time = time.time() - start_time
        
        # Evaluate results
        results = self.evaluate_predictions(predictions, y_test, class_names)
        results['total_inference_time'] = total_time
        results['avg_time_per_sample'] = total_time / len(X_test)
        results['throughput_samples_per_second'] = len(X_test) / total_time
        
        return results
    
    def close(self):
        """Close the HTTP client"""
        self.httpx_client.close()


# If you do not have /tmp/jwt and/or are running the code outside of
# Cloudera AI workbench, modify this function as appropriate
# See https://docs.cloudera.com/machine-learning/cloud/ai-inference/topics/ml-caii-authentication.html
# for the different ways you can get authentication tokens
def load_token(token_path: str = "/tmp/jwt") -> str:
    """Load CDP token from file"""
    try:
        with open(token_path, 'r') as f:
            token_data = json.load(f)
            return token_data["access_token"]
    except Exception as e:
        print(f"Error loading token: {e}")
        raise


def main():
    """Main entry point for the batch inference pipeline"""
    # Configuration
    # Customize these based on your environment
    BASE_URL = f'https://{CAII_DOMAIN}/namespaces/serving-default/endpoints/{ENDPOINT_NAME}'
    MODEL_NAME = f'{MODEL_ID}'
    
    try:
        # Load token
        cdp_token = load_token()
        
        # Initialize inference client
        inference_client = TritonBatchInference(BASE_URL, MODEL_NAME, cdp_token)
        
        # Run the full pipeline
        results = inference_client.run_full_inference_pipeline()
        
        # Display results
        print("\n" + "=" * 50)
        print("INFERENCE RESULTS")
        print("=" * 50)
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Total Samples: {results['total_samples']}")
            print(f"Correct Predictions: {results['correct_predictions']}")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Total Inference Time: {results['total_inference_time']:.3f}s")
            print(f"Average Time per Sample: {results['avg_time_per_sample']:.6f}s")
            print(f"Throughput: {results['throughput_samples_per_second']:.2f} samples/second")
            
            print("\nClassification Report:")
            print(results['classification_report'])
        
        # Close client
        inference_client.close()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())