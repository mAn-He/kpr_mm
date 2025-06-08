import torch
from PIL import Image
import os.path as osp

class ImageCaptioner:
    """Image captioning class that uses LLaVA Mini for image-to-text generation."""
    
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        """Initialize the LLaVA Mini captioning model.
        
        Args:
            model_name: Name of the pre-trained LLaVA model to use.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Import here to avoid dependencies if captioning is not used
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            
            print(f"Loading LLaVA model: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
            print("LLaVA model loaded successfully")
        except ImportError:
            print("Warning: transformers library not found. Install it to use LLaVA captioning.")
            self.processor = None
            self.model = None
    
    def generate_caption(self, image_path):
        """Generate a caption for the given image using LLaVA Mini.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            String caption.
        """
        if self.model is None:
            return "LLaVA model not available"
        
        try:
            # Load and process the image
            image = Image.open(image_path).convert('RGB')
            
            # Create prompt for LLaVA
            prompt = "Describe this person in detail, including clothing, appearance, and position."
            
            # Process inputs
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            # Generate text
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    num_beams=3
                )
            
            # Decode the generated text
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract only the model's response (removing the prompt)
            caption = caption.split(prompt)[-1].strip()
            return caption
            
        except Exception as e:
            print(f"Error generating caption with LLaVA for {image_path}: {e}")
            return f"Error generating caption: {str(e)}"