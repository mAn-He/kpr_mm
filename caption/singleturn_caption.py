"""
Caption images in a folder using Hugging Face‑converted **LLaVA‑1.5 HF** checkpoints
───────────────────────────────────────────────────────────────────────────────
* Works on CUDA or CPU and optionally supports 8‑bit quantisation.
* Generates descriptions for Re‑ID in a JSON list of `{image, description}` objects.
* Supports multi-turn conversation for same ID/class images.
* Includes clothing and colors but excludes accessories and background elements.
"""

import argparse
import gc
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional LLaVA constants (only used for the <image> placeholder)
# ---------------------------------------------------------------------------
try:
    from llava.constants import DEFAULT_IMAGE_TOKEN  # noqa: F401
except ImportError:
    print("Warning: llava.constants.DEFAULT_IMAGE_TOKEN not found – falling back to '<image>'.")
    DEFAULT_IMAGE_TOKEN = "<image>"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

AVAILABLE_MODELS = {
    "llava-7b": "llava-hf/llava-1.5-7b-hf",
    "llava-13b": "llava-hf/llava-1.5-13b-hf",
}

DEFAULT_PROMPT = (
    "Analyze this person as if viewing them from multiple angles simultaneously. "
    "Describe the individual for re-identification including: "
    "1) Their complete outfit from top to bottom with exact colors and style details - "
    "imagine how each piece would look from front, side, and back "
    "2) Their physical characteristics – height category, body shape, limb/torso proportions, "
    "shoulder and hip width, movement patterns suggested by their posture, gait style, "
    "and any permanent anatomical markers visible from any angle "
    "3) Most distinctive visual features that would help identify them even if they turned around. "
    "Create a comprehensive description that would work from any viewing angle. "
    "**Do NOT mention accessories (glasses, jewelry, bags, etc.) or background elements.** "
    "Also, do not use phrases like 'the person' - start directly with the description."
)

# FOLLOW_UP_PROMPT = (
#     "Here is another image of the same individual. Based on this new angle/pose, can you refine or add to your "
#     "previous description? Include clothing details with colors and physical characteristics you can now see better. "
#     "**Do NOT mention accessories (glasses, jewelry, bags, etc.) or background elements.** "
#     "Also, don't start with phrases like 'In the new image' or 'The person is' - continue directly with the description.\n\n"
#     "Your previous description was: {previous_description}"
# )

DEFAULT_IMAGE_FOLDER = (
    # "/scratch/ghtmd9277/keypoint_promptable_reidentification/Occluded_Duke/"
    "/scratch/ghtmd9277/keypoint_promptable_reidentification/Market-1501-v15.09.15/"
    # "bounding_box_train"
    "query"
)

# ---------------------------------------------------------------------------
# Forbidden vocab & regex helpers
# ---------------------------------------------------------------------------
ACCESSORY_WORDS = [
    "glasses", "sunglasses", "earring", "necklace", "bracelet", "watch", "ring", "jewelry",
    "hat", "cap", "beanie", "scarf", "tie", "bow tie", "purse", "bag", "backpack", "handbag",
    "umbrella", "wallet", "phone", "headphone", "earphone", "camera", "book",
]
BACKGROUND_WORDS = [
    "background", "wall", "floor", "ceiling", "building", "street", "road", "sidewalk", 
    "tree", "plant", "car", "vehicle", "bike", "bicycle", "bench", "chair", "table", 
    "door", "window", "sign", "store", "shop", "mall", "crowd",
]
FORBIDDEN_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(ACCESSORY_WORDS + BACKGROUND_WORDS)
    + r")\b",
    re.IGNORECASE,
)

# Patterns to clean up the text
INTRO_PHRASES_PATTERN = re.compile(
    r"^\s*(In the( new)? image,?|The (person|individual|subject)( in the( new)? image)? (is|appears to be))",
    re.IGNORECASE
)
EMPTY_SUBJECT_PATTERN = re.compile(r"The\s+is", re.IGNORECASE)
DOUBLE_SPACES_PATTERN = re.compile(r"\s{2,}")

def cleanse_description(text: str) -> str:
    """Clean up the description by removing forbidden words and fixing text issues."""
    # Remove sentences with forbidden content
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        if not s.strip():
            continue
        if FORBIDDEN_PATTERN.search(s):
            # redact forbidden words
            redacted = FORBIDDEN_PATTERN.sub("", s).strip()
            if len(redacted.split()) >= 4:
                kept.append(redacted)
        else:
            kept.append(s.strip())
    
    text = " ".join(kept).strip()
    
    # Remove intro phrases like "In the new image, the person is"
    text = INTRO_PHRASES_PATTERN.sub("", text).strip()
    
    # Fix "The is" to "This is" (assuming it should be "This is")
    text = EMPTY_SUBJECT_PATTERN.sub("This is", text)
    
    # Remove double spaces
    text = DOUBLE_SPACES_PATTERN.sub(" ", text)
    
    # Capitalize first letter
    if text and len(text) > 0:
        text = text[0].upper() + text[1:]
    
    return text

# ---------------------------------------------------------------------------
# ID extraction helper
# ---------------------------------------------------------------------------
def extract_person_id(filename: str) -> str:
    """Extract person ID from Market-1501 filename format."""
    # Market-1501 format: 0002_c1s1_000451_03.jpg -> ID = 0002
    match = re.match(r"(\d{4})_", filename)
    if match:
        return match.group(1)
    return "unknown"

# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------

def check_and_install_dependencies() -> bool:
    required = {
        "transformers": "transformers>=4.47.0",
        "tokenizers": "tokenizers>=0.15.0",
        "accelerate": "accelerate>=0.21.0",
        "timm": "timm>=0.9.12",
        "Pillow": "pillow",
        "tqdm": "tqdm",
        "packaging": "packaging",
    }
    if any("--quantize" in arg for arg in sys.argv):
        required["bitsandbytes"] = "bitsandbytes"

    missing = []
    for module_name, pip_spec in required.items():
        try:
            base = module_name.split("==")[0].split(">=")[0]
            importlib.import_module("PIL.Image" if base == "Pillow" else base)
        except ImportError:
            missing.append(pip_spec)

    if missing:
        print(f"Installing/upgrading missing packages: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *missing])
        except subprocess.CalledProcessError:
            print("✘ Failed to install some dependencies – please install manually.")
            return False
    return True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Vision–Language captioning (with clothing and colors)")
    p.add_argument("--model_key", choices=list(AVAILABLE_MODELS.keys()), default="llava-7b")
    p.add_argument("--image_folder", default=DEFAULT_IMAGE_FOLDER)
    p.add_argument("--num_images", type=int, default=5)
    p.add_argument("--output_file", default="market_1501_query_singleturn.json")
    p.add_argument("--quantize", action="store_true", help="Enable 8‑bit quantisation (CUDA only)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--disable_multiturn", action="store_true", help="Disable multi-turn functionality")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_llava_hf_model(model_key: str, quantize: bool):
    print(f"Loading LLaVA model: {AVAILABLE_MODELS[model_key]}")
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    q_cfg = BitsAndBytesConfig(load_in_8bit=True) if quantize and DEVICE == "cuda" else None

    # Load processor (slow tokenizer avoids JSON parse issue)
    processor = AutoProcessor.from_pretrained(
        AVAILABLE_MODELS[model_key], trust_remote_code=True, use_fast=False
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        AVAILABLE_MODELS[model_key],
        torch_dtype=DTYPE,
        device_map="auto",
        quantization_config=q_cfg,
        trust_remote_code=True,
    ).eval()
    return model, processor

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_description(model, processor, image_pil: Image.Image, prompt_text: str) -> str:
    formatted_prompt = f"USER: {DEFAULT_IMAGE_TOKEN}\n{prompt_text}\nASSISTANT:"
    inputs = processor(text=formatted_prompt, images=image_pil, return_tensors="pt")
    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device, dtype=DTYPE if torch.is_floating_point(v) else None) for k, v in inputs.items()}
    out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    raw = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return cleanse_description(raw)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# def main():
#     if not check_and_install_dependencies():
#         sys.exit(1)

#     args = parse_args()
#     model, processor = load_llava_hf_model(args.model_key, args.quantize)

#     image_paths = sorted(Path(args.image_folder).glob("*.jpg"))
#     if args.num_images > 0:
#         image_paths = image_paths[:]
#         # [:args.num_images]
        
#     if not image_paths:
#         sys.exit(f"✘ No .jpg images found in {args.image_folder}")

#     # Group images by person ID
#     images_by_id = defaultdict(list)
#     for img_path in image_paths:
#         person_id = extract_person_id(img_path.name)
#         images_by_id[person_id].append(img_path)
    
#     results = []
#     id_descriptions = {}  # Store the latest description for each ID
    
#     # Process images by ID groups
#     for person_id, id_images in tqdm(images_by_id.items(), desc="Processing IDs"):
#         for i, img_path in enumerate(id_images):
#             try:
#                 img = Image.open(img_path).convert("RGB")
#             except Exception as e:
#                 results.append({"image": img_path.name, "description": f"Error loading image: {e}"})
#                 continue

#             try:
#                 # First image of this ID - use standard prompt
#                 if i == 0 or args.disable_multiturn:
#                     desc = generate_description(model, processor, img, args.prompt)
#                 else:
#                     # Follow-up image - use previous description in the prompt
#                     follow_up = FOLLOW_UP_PROMPT.format(previous_description=id_descriptions[person_id])
#                     desc = generate_description(model, processor, img, follow_up)
                
#                 if len(desc.split()) < 4:
#                     desc = "No valid description could be generated."
                
#                 # Store the description for this ID for future references
#                 id_descriptions[person_id] = desc
                
#             except Exception as e:
#                 desc = f"Error during generation: {e}"

#             results.append({
#                 "image": img_path.name, 
#                 "description": desc,
#                 "person_id": person_id,
#                 "is_follow_up": i > 0 and not args.disable_multiturn
#             })
            
#             tqdm.write(f"{img_path.name} (ID: {person_id}) → {desc[:90]}…")
#             gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

#     Path(args.output_file).write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
#     print(f"✓ Saved {len(results)} descriptions → {args.output_file}")

def main():
    if not check_and_install_dependencies():
        sys.exit(1)

    args = parse_args()
    model, processor = load_llava_hf_model(args.model_key, args.quantize)

    image_paths = sorted(Path(args.image_folder).glob("*.jpg"))
    if args.num_images > 0:
        image_paths = image_paths[:]
        # [:args.num_images]
        
    if not image_paths:
        sys.exit(f"✘ No .jpg images found in {args.image_folder}")

    # Group images by person ID
    images_by_id = defaultdict(list)
    for img_path in image_paths:
        person_id = extract_person_id(img_path.name)
        images_by_id[person_id].append(img_path)
    
    results = []
    
    # Process images by ID groups
    for person_id, id_images in tqdm(images_by_id.items(), desc="Processing IDs"):
        for i, img_path in enumerate(id_images):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                results.append({"image": img_path.name, "description": f"Error loading image: {e}"})
                continue

            try:
                # Single-turn: 모든 이미지에 동일한 프롬프트 사용
                desc = generate_description(model, processor, img, args.prompt)
                
                if len(desc.split()) < 4:
                    desc = "No valid description could be generated."
                
            except Exception as e:
                desc = f"Error during generation: {e}"

            results.append({
                "image": img_path.name, 
                "description": desc,
                "person_id": person_id
            })
            
            tqdm.write(f"{img_path.name} (ID: {person_id}) → {desc[:90]}…")
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    Path(args.output_file).write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
    print(f"✓ Saved {len(results)} descriptions → {args.output_file}")


if __name__ == "__main__":
    main()
