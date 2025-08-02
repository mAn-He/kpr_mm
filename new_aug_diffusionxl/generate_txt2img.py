import os
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline
import torch
from accelerate import Accelerator
import gc
# from tqdm import tqdm # Removed tqdm

# HF_CACHE_DIR = "/scratch/ghtmd9277/huggingface_cache"
# os.makedirs(HF_CACHE_DIR, exist_ok=True)

# 환경 변수 설정
# os.environ['HF_HOME'] = HF_CACHE_DIR
# os.environ['TRANSFORMERS_CACHE'] = os.path.join(HF_CACHE_DIR, 'transformers')
# os.environ['DIFFUSERS_CACHE'] = os.path.join(HF_CACHE_DIR, 'diffusers')
# os.environ['HF_DATASETS_CACHE'] = os.path.join(HF_CACHE_DIR, 'datasets')

class ImageAugmentor:
    def __init__(self, accelerator, output_dir="./augmented_images_test"):
        self.accelerator = accelerator
        
        # Initialize SDXL Inpainting pipeline with accelerator
        print("Initializing SDXL model...")
        self.inpaint_model = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        # No need to prepare model with accelerator for single GPU test
        # self.inpaint_model = accelerator.prepare(self.inpaint_model) 
        self.inpaint_model.to(accelerator.device) # Move model to the correct device
        
        # Memory optimization
        self.inpaint_model.enable_attention_slicing()
        self.inpaint_model.enable_vae_slicing()

        # Ensure reproducibility
        torch.manual_seed(1234)
        np.random.seed(1234)

        # Result directories
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.skipped_files = [] # Initialize list to track skipped files

        # Campus environment prompts
        self.prompts = [
            "University campus pathway, brick buildings background, sunny daylight, distant pedestrians.",
            "Paved walkway through green campus lawn, lined with trees, clear blue sky.",
            "Modern university plaza, concrete ground, glass building facade distance, students walking.",
            "Campus sidewalk beside building with large windows, trees bordering, bright daytime.",
            "Outdoor university campus scene, mix of architecture and nature, bright daylight."
        ]

    def augment_images(self, image_files, input_dir, mask_dir):
        """Process multiple images""" # Removed progress bar description
        self.mask_dir = mask_dir
        self.skipped_files = [] # Reset skipped files list for this run
        
        # Removed tqdm initialization
        
        print(f"Processing {len(image_files)} images...")
        for image_file in image_files:
            try:
                img_path = os.path.join(input_dir, image_file)
                print(f"Processing: {img_path}")
                skipped = self.augment_and_save_image(img_path)
                if skipped:
                    self.skipped_files.append(image_file) # Add to skipped list if mask was missing

                # Removed tqdm update
                
                # Memory cleanup (Removed explicit calls for potential speedup)
                # gc.collect()
                # torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        # Removed tqdm close
        print("Finished processing images.")


    def augment_and_save_image(self, img_path):
        """Process single image. Returns True if skipped due to missing mask, False otherwise."""
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image {img_path}")
            return False # Not skipped due to mask

        original_filename = os.path.splitext(os.path.basename(img_path))[0]

        # Select and validate mask
        mask_path = self._select_mask(original_filename)
        if not mask_path:
            print(f"Skipping {original_filename}: No valid mask found")
            return True # Skipped due to mask

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        inverted_mask = cv2.bitwise_not(mask)

        # Generate images for each prompt
        for idx, prompt in enumerate(self.prompts):
            try:
                # print(f"  Applying prompt {idx+1}: {prompt}")
                inpainted_image = self._apply_inpainting(img, inverted_mask, prompt, original_size=img.shape[:2])
                if inpainted_image is not None and not self._is_black_image(inpainted_image):
                    self._save_image(inpainted_image, original_filename, idx + 1)
                else:
                    print(f"  Skipping {original_filename} with prompt {idx+1}: Invalid result")
            except Exception as e:
                 print(f"  Error during inpainting for {original_filename} with prompt {idx+1}: {e}")
        
        return False # Not skipped due to mask

    def _select_mask(self, original_filename):
        """Select and validate mask file"""
        mask_path = os.path.join(self.mask_dir, f"{original_filename}_mask.png")
        
        if not os.path.exists(mask_path):
            print(f"No mask found for {original_filename} at {mask_path}")
            return None
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or self._is_black_image(mask):
            print(f"Invalid mask for {original_filename}")
            return None
            
        return mask_path

    def _apply_inpainting(self, img, mask, prompt, original_size):
        """Apply SDXL inpainting with optimized parameters"""
        # Convert to RGB
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)

        try:
            # No autocast needed for single GPU
            # with self.accelerator.autocast(): 
            # Perform inpainting with optimized parameters
            result = self.inpaint_model(
                prompt=prompt,
                image=img_pil,
                mask_image=mask_pil,
                num_inference_steps=20,  # Reduced steps
                guidance_scale=7.5,
                strength=0.7,  # Better instance preservation
            ).images[0]

            # Convert back to BGR and resize
            result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            result_resized = cv2.resize(
                result_cv,
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
            
            return result_resized
            
        except Exception as e:
            print(f"Inpainting error: {e}")
            return None

    def _is_black_image(self, img):
        """Check for black image"""
        return np.all(img == 0)

    def _save_image(self, inpainted_img, original_filename, augment_idx):
        """Save augmented image"""
        # Save within a subfolder named after the original image
        # base_path = os.path.join(self.output_dir, original_filename) 
        # os.makedirs(base_path, exist_ok=True)
        # save_path = os.path.join(base_path, f"{original_filename}_augmented_{augment_idx}.png")
        
        # Save directly into the output directory for simplicity in testing
        save_path = os.path.join(self.output_dir, f"{original_filename}_augmented_{augment_idx}.jpg")
        cv2.imwrite(save_path, inpainted_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90]) # Save as JPG
        
        print(f"Saved: {save_path}")

def main():
    # Initialize accelerator (still useful for device placement)
    accelerator = Accelerator(
        mixed_precision='fp16' 
        # Removed gradient_accumulation_steps and dynamo_backend for simplicity
    )

    # Paths
    input_dir = "./Market-1501-v15.09.15/bounding_box_train"
    mask_dir = "./Market-1501-v15.09.15/market_mask"
    output_dir = "./Market-1501-v15.09.15/aug_market_inpainting_5prompts" # Changed output dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get image files
    all_image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    # Sort for deterministic distribution
    all_image_files.sort()
    
    # Select only the first 5 images for testing
    image_files_to_process = all_image_files

    # Removed multi-GPU distribution logic
    print(f"Total images found: {len(all_image_files)}")
    print(f"Processing first 5 images: {image_files_to_process}")

    try:
        # Create augmentor and process images
        augmentor = ImageAugmentor(accelerator, output_dir=output_dir)
        # Pass only the selected 5 images
        augmentor.augment_images(image_files_to_process, input_dir, mask_dir) 

        # No need to wait for other processes
        # accelerator.wait_for_everyone() 

        print("Test augmentation complete.")
        
        # Print skipped files
        if augmentor.skipped_files:
            print("\nThe following files were skipped due to missing or invalid masks:")
            for skipped_file in augmentor.skipped_files:
                print(f"- {skipped_file}")
            
    except Exception as e:
        print(f"Error in main process: {e}")
        raise e

if __name__ == "__main__":
    main()
