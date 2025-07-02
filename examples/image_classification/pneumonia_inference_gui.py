from open_inference.openapi.client import OpenInferenceClient, InferenceRequest
import httpx
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import pandas as pd
import time
from typing import Optional, Dict, Any, Tuple
import io
import base64
import os


class PneumoniaInferenceClient:
    """Client for pneumonia classification inference using Open Inference protocol"""
    
    def __init__(self, base_url: str, model_name: str, token: str):
        self.base_url = base_url
        self.model_name = model_name
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        self.httpx_client = httpx.Client(headers=self.headers)
        self.client = OpenInferenceClient(base_url=base_url, httpx_client=self.httpx_client)
        
        # Initialize image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])  # Grayscale normalization
        ])
        
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess uploaded image for inference"""
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Apply transforms
            tensor_image = self.transform(image)
            
            # Add batch dimension and convert to numpy
            tensor_image = tensor_image.unsqueeze(0)  # Shape: (1, 1, 224, 224)
            numpy_image = tensor_image.numpy()
            
            return numpy_image
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    def check_server_status(self) -> Tuple[bool, str]:
        """Check if the server is ready and get model metadata"""
        try:
            # Check server readiness
            self.client.check_server_readiness()
            status_msg = "✓ Server is ready"
            
            # Get model metadata
            metadata = self.client.read_model_metadata(self.model_name)
            metadata_dict = json.loads(metadata.json())
            
            model_info = f"""
Model Information:
- Name: {metadata_dict.get('name', 'Unknown')}
- Platform: {metadata_dict.get('platform', 'Unknown')}
- Inputs: {len(metadata_dict.get('inputs', []))}
- Outputs: {len(metadata_dict.get('outputs', []))}
"""
            
            return True, status_msg + model_info
            
        except Exception as e:
            return False, f"Error checking server status: {str(e)}"
    
    def run_inference(self, image: Image.Image) -> Dict[str, Any]:
        """Run inference on a single chest X-ray image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Create inference request
            inference_request = InferenceRequest(
                inputs=[{
                    "name": "input",  # Adjust based on your model's input name
                    "shape": list(processed_image.shape),
                    "datatype": "FP32",
                    "data": processed_image.flatten().tolist()
                }],
            )
            
            # Run inference
            start_time = time.time()
            response = self.client.model_infer(self.model_name, request=inference_request)
            inference_time = time.time() - start_time
            
            # Extract predictions from response
            response_dict = json.loads(response.json())
            output_data = response_dict['outputs'][0]['data']
            
            # Process predictions
            predictions = np.array(output_data).reshape(1, -1)  # Shape: (1, 2)
            probabilities = torch.softmax(torch.tensor(predictions), dim=1).numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            # Map predictions to class names
            class_names = ['NORMAL', 'PNEUMONIA']
            predicted_label = class_names[predicted_class]
            
            return {
                'prediction': predicted_label,
                'confidence': float(confidence),
                'probabilities': {
                    'NORMAL': float(probabilities[0]),
                    'PNEUMONIA': float(probabilities[1])
                },
                'inference_time': inference_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'probabilities': {'NORMAL': 0.0, 'PNEUMONIA': 0.0},
                'inference_time': 0.0,
                'error': str(e),
                'success': False
            }
    
    def close(self):
        """Close the HTTP client"""
        self.httpx_client.close()


# Global client instance
client = None


def load_token(token_path: str = "/tmp/jwt") -> str:
    """Load CDP token from file"""
    try:
        with open(token_path, 'r') as f:
            token_data = json.load(f)
            return token_data["access_token"]
    except Exception as e:
        # For demo purposes, return a placeholder token
        # In production, this should be properly configured
        return "your-auth-token-here"


def initialize_client(base_url: str, model_name: str) -> Tuple[bool, str]:
    """Initialize the inference client"""
    global client
    
    try:
        token = load_token()
        client = PneumoniaInferenceClient(base_url, model_name, token)
        
        # Check server status
        is_ready, status_msg = client.check_server_status()
        
        if is_ready:
            return True, f"✅ Client initialized successfully!\n{status_msg}"
        else:
            return False, f"❌ Server not ready: {status_msg}"
            
    except Exception as e:
        return False, f"❌ Failed to initialize client: {str(e)}"


def predict_pneumonia(image: Image.Image, base_url: str, model_name: str) -> Tuple[str, str, Dict, str]:
    """Main prediction function for Gradio interface"""
    global client
    
    # Validate inputs
    if image is None:
        empty_chart = pd.DataFrame({"Class": [], "Probability": []})
        return "❌ No image uploaded", "Please upload a chest X-ray image", empty_chart, ""
    
    if not base_url or not model_name:
        empty_chart = pd.DataFrame({"Class": [], "Probability": []})
        return "❌ Configuration Error", "Please provide base URL and model name", empty_chart, ""
    
    try:
        # Initialize client if not already done
        if client is None:
            success, msg = initialize_client(base_url, model_name)
            if not success:
                empty_chart = pd.DataFrame({"Class": [], "Probability": []})
                return "❌ Connection Failed", msg, empty_chart, ""
        
        # Run inference
        result = client.run_inference(image)
        
        if result['success']:
            # Format results
            prediction_text = f"🔍 **Prediction: {result['prediction']}**"
            confidence_text = f"📊 **Confidence: {result['confidence']:.2%}**"
            time_text = f"⏱️ **Inference Time: {result['inference_time']:.3f}s**"
            
            status_message = f"{prediction_text}\n{confidence_text}\n{time_text}"
            
            # Create probability chart data
            prob_chart = pd.DataFrame({
                "Class": ["Normal", "Pneumonia"],
                "Probability": [result['probabilities']['NORMAL'], result['probabilities']['PNEUMONIA']]
            })
            
            # Create detailed results
            detailed_results = f"""
## Detailed Results

**Classification:** {result['prediction']}
**Confidence Score:** {result['confidence']:.4f}

**Class Probabilities:**
- Normal: {result['probabilities']['NORMAL']:.4f} ({result['probabilities']['NORMAL']*100:.2f}%)
- Pneumonia: {result['probabilities']['PNEUMONIA']:.4f} ({result['probabilities']['PNEUMONIA']*100:.2f}%)

**Performance:**
- Inference Time: {result['inference_time']:.3f} seconds

---
⚠️ **Disclaimer:** This is a demonstration model and should not be used for actual medical diagnosis. 
Always consult with qualified healthcare professionals for medical decisions.
"""
            
            return status_message, detailed_results, prob_chart, "✅ Inference completed successfully"
            
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            empty_chart = pd.DataFrame({"Class": [], "Probability": []})
            return f"❌ Inference Failed", f"Error: {error_msg}", empty_chart, "❌ Inference failed"
            
    except Exception as e:
        empty_chart = pd.DataFrame({"Class": [], "Probability": []})
        return f"❌ Unexpected Error", f"An unexpected error occurred: {str(e)}", empty_chart, "❌ Unexpected error"


def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .error {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    """
    
    with gr.Blocks(
        title="Pneumonia Classification - Chest X-Ray Analysis",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:
        
        gr.Markdown("""
        # 🫁 Pneumonia Classification System
        
        Upload a chest X-ray image to get an AI-powered assessment for pneumonia detection.
        
        **Instructions:**
        1. Configure your model endpoint settings below
        2. Upload a chest X-ray image (JPEG, PNG formats supported)
        3. Click "Analyze X-Ray" to get the prediction
        
        ⚠️ **Medical Disclaimer:** This tool is for demonstration purposes only and should not be used for actual medical diagnosis.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔧 Configuration")
                
                base_url_input = gr.Textbox(
                    label="Model Endpoint URL",
                    placeholder="https://your-inference-endpoint.com/v2/models",
                    value="",
                    info="Base URL for your model inference endpoint"
                )
                
                model_name_input = gr.Textbox(
                    label="Model Name",
                    placeholder="pneumonia_onnx_classifier",
                    value="pneumonia_onnx_classifier",
                    info="Name of the deployed model"
                )
                
                test_connection_btn = gr.Button("🔗 Test Connection", variant="secondary")
                connection_status = gr.Textbox(
                    label="Connection Status",
                    interactive=False,
                    max_lines=5
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 📸 Image Upload")
                
                image_input = gr.Image(
                    label="Upload Chest X-Ray Image",
                    type="pil",
                    height=300
                )
                
                analyze_btn = gr.Button("🔍 Analyze X-Ray", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Results")
                
                prediction_output = gr.Textbox(
                    label="Prediction Result",
                    interactive=False,
                    max_lines=3
                )
                
                probability_chart = gr.BarPlot(
                    x="Class",
                    y="Probability",
                    title="Classification Probabilities",
                    x_title="Diagnosis Class",
                    y_title="Probability Score",
                    height=300
                )
                
            with gr.Column():
                gr.Markdown("### 📋 Detailed Analysis")
                
                detailed_output = gr.Markdown(
                    value="Upload an image and click 'Analyze X-Ray' to see detailed results."
                )
        
        with gr.Row():
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False
            )
        
        # Event handlers
        test_connection_btn.click(
            fn=lambda url, name: initialize_client(url, name)[1],
            inputs=[base_url_input, model_name_input],
            outputs=[connection_status]
        )
        
        analyze_btn.click(
            fn=predict_pneumonia,
            inputs=[image_input, base_url_input, model_name_input],
            outputs=[prediction_output, detailed_output, probability_chart, status_output]
        )
        
        # Example images section
        gr.Markdown("""
        ### 📝 Usage Tips
        
        - **Image Quality:** Use clear, high-resolution chest X-ray images for best results
        - **Format:** JPEG, PNG, and other common image formats are supported
        - **Preprocessing:** Images are automatically resized and normalized for the model
        - **Privacy:** Images are processed locally and not stored permanently
        
        ### 🔧 Configuration Help
        
        - **Endpoint URL:** Should point to your Triton Inference Server or similar OpenAPI-compatible endpoint
        - **Model Name:** Must match the exact name of your deployed ONNX model
        - **Authentication:** Ensure your token file is properly configured at `/tmp/jwt`
        """)
    
    return interface


def main():
    """Main entry point"""
    # Create and launch the Gradio interface
    interface = create_gradio_interface()
    
    # Launch with configuration
    interface.launch(
        server_name="localhost",  # Allow external access
        server_port=int(os.environ.get('CDSW_APP_PORT')),
        share=False,            # Set to True to create public link
        debug=True,             # Enable debug mode
        show_error=True         # Show error messages
    )


if __name__ == "__main__":
    main() 