import json
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import pandas as pd
import time
from typing import Dict, Any, Tuple
import random


class MockPneumoniaInferenceClient:
    """Mock client for pneumonia classification inference for demo purposes"""
    
    def __init__(self, base_url: str, model_name: str, token: str):
        self.base_url = base_url
        self.model_name = model_name
        self.token = token
        
        # Initialize image preprocessing transforms (same as real model)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])  # Grayscale normalization
        ])
        
        print(f"Mock client initialized for model: {model_name}")
        
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
        """Mock server status check"""
        model_info = f"""
Model Information:
- Name: {self.model_name}
- Platform: ONNX Runtime (Mock)
- Inputs: 1 (chest X-ray image)
- Outputs: 1 (classification probabilities)
- Status: Demo/Mock Mode
"""
        return True, "✓ Mock server is ready" + model_info
    
    def run_inference(self, image: Image.Image) -> Dict[str, Any]:
        """Run mock inference on a single chest X-ray image"""
        try:
            # Preprocess image (for validation)
            processed_image = self.preprocess_image(image)
            
            # Simulate inference time
            inference_time = random.uniform(0.1, 0.5)
            time.sleep(inference_time)
            
            # Generate realistic mock predictions
            # Simulate some logic based on image characteristics
            img_array = np.array(image.convert('L'))
            brightness = np.mean(img_array)
            
            # Mock logic: darker images slightly more likely to be classified as pneumonia
            if brightness < 128:
                normal_prob = random.uniform(0.2, 0.6)
            else:
                normal_prob = random.uniform(0.4, 0.8)
                
            pneumonia_prob = 1.0 - normal_prob
            
            probabilities = np.array([normal_prob, pneumonia_prob])
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
        """Mock close method"""
        print("Mock client closed")


# Global client instance
client = None


def initialize_mock_client(base_url: str, model_name: str) -> Tuple[bool, str]:
    """Initialize the mock inference client"""
    global client
    
    try:
        client = MockPneumoniaInferenceClient(base_url, model_name, "mock_token")
        
        # Check server status
        is_ready, status_msg = client.check_server_status()
        
        if is_ready:
            return True, f"✅ Mock client initialized successfully!\n{status_msg}"
        else:
            return False, f"❌ Mock server not ready: {status_msg}"
            
    except Exception as e:
        return False, f"❌ Failed to initialize mock client: {str(e)}"


def predict_pneumonia_demo(image: Image.Image, base_url: str, model_name: str) -> Tuple[str, str, Dict, str]:
    """Demo prediction function for Gradio interface"""
    global client
    
    # Validate inputs
    if image is None:
        return "❌ No image uploaded", "Please upload a chest X-ray image", {}, ""
    
    # Use default values if not provided
    if not base_url:
        base_url = "http://demo-endpoint.com"
    if not model_name:
        model_name = "pneumonia_demo_model"
    
    try:
        # Initialize client if not already done
        if client is None:
            success, msg = initialize_mock_client(base_url, model_name)
            if not success:
                return "❌ Connection Failed", msg, {}, ""
        
        # Run inference
        result = client.run_inference(image)
        
        if result['success']:
            # Format results
            prediction_text = f"🔍 **Prediction: {result['prediction']}**"
            confidence_text = f"📊 **Confidence: {result['confidence']:.2%}**"
            time_text = f"⏱️ **Inference Time: {result['inference_time']:.3f}s**"
            
            status_message = f"{prediction_text}\n{confidence_text}\n{time_text}"
            
            # Create probability chart data for Gradio
            prob_data = pd.DataFrame({
                "Class": ["Normal", "Pneumonia"],
                "Probability": [result['probabilities']['NORMAL'], result['probabilities']['PNEUMONIA']]
            })
            
            # Create detailed results
            detailed_results = f"""
## Detailed Results (Demo Mode)

**Classification:** {result['prediction']}
**Confidence Score:** {result['confidence']:.4f}

**Class Probabilities:**
- Normal: {result['probabilities']['NORMAL']:.4f} ({result['probabilities']['NORMAL']*100:.2f}%)
- Pneumonia: {result['probabilities']['PNEUMONIA']:.4f} ({result['probabilities']['PNEUMONIA']*100:.2f}%)

**Performance:**
- Inference Time: {result['inference_time']:.3f} seconds

---
🧪 **Demo Mode:** This is a demonstration with simulated predictions. 
In production, connect to your actual deployed pneumonia classification model.

⚠️ **Medical Disclaimer:** This tool is for demonstration purposes only and should not be used for actual medical diagnosis. 
Always consult with qualified healthcare professionals for medical decisions.
"""
            
            return status_message, detailed_results, prob_data, "✅ Demo inference completed successfully"
            
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            return f"❌ Inference Failed", f"Error: {error_msg}", [], "❌ Inference failed"
            
    except Exception as e:
        return f"❌ Unexpected Error", f"An unexpected error occurred: {str(e)}", [], "❌ Unexpected error"


def create_pneumonia_demo_interface():
    """Create the demo Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .demo-banner {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    """
    
    with gr.Blocks(
        title="Pneumonia Classification Demo - Chest X-Ray Analysis",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:
        
        gr.HTML("""
        <div class="demo-banner">
            <h1>🫁 Pneumonia Classification Demo</h1>
            <p>Interactive demonstration of AI-powered chest X-ray analysis</p>
            <p><strong>🧪 Demo Mode:</strong> Using simulated inference for demonstration purposes</p>
        </div>
        """)
        
        gr.Markdown("""
        ## How to Use This Demo
        
        1. **Upload Image:** Click on the image upload area and select a chest X-ray image
        2. **Configure Settings:** (Optional) Modify the endpoint settings for production use
        3. **Analyze:** Click "Analyze X-Ray" to get AI predictions
        4. **Review Results:** See the classification results and confidence scores
        
        ⚠️ **Important:** This demo uses simulated predictions for demonstration purposes.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📸 Upload Chest X-Ray Image")
                
                image_input = gr.Image(
                    label="Drop your chest X-ray image here or click to upload",
                    type="pil",
                    height=400
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze X-Ray", 
                    variant="primary", 
                    size="lg"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration (Optional)")
                
                gr.Markdown("*For production deployment:*")
                
                base_url_input = gr.Textbox(
                    label="Model Endpoint URL",
                    placeholder="http://your-inference-endpoint.com",
                    value="http://demo-endpoint.com",
                    info="Leave as default for demo mode"
                )
                
                model_name_input = gr.Textbox(
                    label="Model Name",
                    placeholder="pneumonia_classifier",
                    value="pneumonia_demo_model",
                    info="Leave as default for demo mode"
                )
                
                test_connection_btn = gr.Button("🔗 Test Connection", variant="secondary")
                connection_status = gr.Textbox(
                    label="Connection Status",
                    interactive=False,
                    max_lines=8,
                    value="Click 'Test Connection' to check demo status"
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Classification Results")
                
                prediction_output = gr.Textbox(
                    label="Prediction Summary",
                    interactive=False,
                    max_lines=4,
                    placeholder="Upload an image and click 'Analyze X-Ray' to see results"
                )
                
                probability_chart = gr.BarPlot(
                    x_lim=[0, 1],
                    title="Classification Probabilities",
                    x_title="Probability Score",
                    y_title="Diagnosis Class",
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
        
        # Sample images section
        gr.Markdown("""
        ### 📝 Demo Instructions & Tips
        
        **For Testing:**
        - Upload any chest X-ray image (JPEG, PNG formats)
        - The demo will provide simulated classification results
        - Try different images to see varied predictions
        
        **Image Requirements:**
        - Clear chest X-ray images work best
        - Images are automatically resized and processed
        - Both color and grayscale images are supported
        
        **Production Deployment:**
        - Replace the mock client with the real inference client
        - Configure proper authentication and endpoint URLs
        - Deploy your trained ONNX model to a compatible inference server
        
        ---
        
        ### 🔬 About the AI Model
        
        The production model is a Convolutional Neural Network (CNN) trained specifically for pneumonia detection in chest X-rays:
        
        - **Architecture:** Custom CNN with 4 convolutional blocks
        - **Input:** 224x224 grayscale chest X-ray images  
        - **Output:** Binary classification (Normal vs Pneumonia)
        - **Training:** Kaggle chest X-ray pneumonia dataset
        - **Framework:** PyTorch → ONNX for deployment
        """)
        
        # Event handlers
        test_connection_btn.click(
            fn=lambda url, name: initialize_mock_client(url, name)[1],
            inputs=[base_url_input, model_name_input],
            outputs=[connection_status]
        )
        
        analyze_btn.click(
            fn=predict_pneumonia_demo,
            inputs=[image_input, base_url_input, model_name_input],
            outputs=[prediction_output, detailed_output, probability_chart, status_output]
        )
        
    return interface


def main():
    """Main entry point for the demo"""
    print("🚀 Starting Pneumonia Classification Demo...")
    print("📝 This is a demonstration interface with simulated predictions")
    print("🌐 Access at: http://localhost:7860")
    
    # Create and launch the demo interface
    interface = create_pneumonia_demo_interface()
    
    # Launch with configuration
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        debug=True,             # Enable debug mode
        show_error=True         # Show error messages
    )


if __name__ == "__main__":
    main() 