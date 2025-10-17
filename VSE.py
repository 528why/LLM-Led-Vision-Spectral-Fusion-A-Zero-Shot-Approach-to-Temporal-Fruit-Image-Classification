import os
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Import libraries for GLIP and SAM
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from segment_anything import sam_model_registry, SamPredictor

# Import the library to generate the text prompt
from LLM_as_expert import extract_knowledge

# --- 1. Configuration ---

# GLIP Model Settings
CONFIG_FILE_GLIP = "Path to your GLIP configs(yaml)"
WEIGHT_FILE_GLIP = "Path to your GLIP Model"

# SAM Model Settings
SAM_CHECKPOINT = "Path to your SAM Model"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generate Text Prompt using the LLM_as_expert library
TEXT_PROMPT = extract_knowledge('avocado')
print(f"Using generated text prompt: '{TEXT_PROMPT}'")

# --- Data Path Settings ---
# Please update these paths to match your local directory structure.

INPUT_TRAIN_DIR = 'Path to your original training image dataset'
INPUT_TEST_DIR = 'Path to your original test image dataset'
OUTPUT_TRAIN_DIR = 'Directory where the processed training images will be saved'
OUTPUT_TEST_DIR = 'Directory where the processed test images will be saved'


# --- 2. Model Initialization ---

def initialize_models():
    """Loads and initializes the GLIP and SAM models."""
    print("Initializing GLIP model...")
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(CONFIG_FILE_GLIP)
    cfg.merge_from_list(["MODEL.WEIGHT", WEIGHT_FILE_GLIP])
    cfg.merge_from_list(["MODEL.DEVICE", DEVICE])

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )
    print("GLIP model loaded successfully.")

    print("Initializing SAM model...")
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"SAM checkpoint '{SAM_CHECKPOINT}' not found. Downloading...")
        import wget
        wget.download(f"https://dl.fbaipublicfiles.com/segment_anything/{SAM_CHECKPOINT}")

    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    print("SAM model loaded successfully.")

    return glip_demo, predictor

# --- 3. Helper Functions ---

def resize_and_pad(image_np, target_size=512, pad_color=255):
    """Resizes and pads an image to a target size while maintaining aspect ratio."""
    original_height, original_width = image_np.shape[:2]
    scale = target_size / max(original_width, original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(image_np.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_AREA)

    pad_top = (target_size - new_height) // 2
    pad_bottom = target_size - new_height - pad_top
    pad_left = (target_size - new_width) // 2
    pad_right = target_size - new_width - pad_left

    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[pad_color, pad_color, pad_color]
    )
    return padded_image

def image_processor(image, glip_demo, predictor, caption):
    """Executes the full processing pipeline for a single image."""
    image_np_rgb = np.array(image)
    image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

    top_predictions = glip_demo.inference(image_np_bgr, caption)
    bboxes = top_predictions.bbox

    if bboxes.dim() == 1 and bboxes.shape[0] == 0:
        print("Warning: GLIP did not detect any objects. Returning resized original image.")
        return resize_and_pad(image_np_rgb)

    bbox = np.array(bboxes[0].cpu())
    predictor.set_image(image_np_rgb)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox[None, :],
        multimask_output=False,
    )
    mask = masks[0]

    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    white_background = np.ones_like(image_np_rgb) * 255
    result_image = np.where(mask_3d, image_np_rgb, white_background)
    resized_padded_image = resize_and_pad(result_image)
    return resized_padded_image

def process_dataset(input_dir, output_dir, glip_model, sam_predictor, caption):
    """Iterates through all images in a given directory, processes them, and saves the results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in tqdm(image_files, desc=f"Processing images in {input_dir}"):
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        if os.path.exists(output_path):
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            processed_image_np = image_processor(image, glip_model, sam_predictor, caption)
            processed_image_pil = Image.fromarray(processed_image_np.astype(np.uint8))
            processed_image_pil.save(output_path)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# --- 4. Main Execution Flow ---

if __name__ == "__main__":
    glip_model, sam_predictor = initialize_models()

    print("\n--- Starting to process the training dataset ---")
    process_dataset(INPUT_TRAIN_DIR, OUTPUT_TRAIN_DIR, glip_model, sam_predictor, TEXT_PROMPT)

    print("\n--- Starting to process the test dataset ---")
    process_dataset(INPUT_TEST_DIR, OUTPUT_TEST_DIR, glip_model, sam_predictor, TEXT_PROMPT)

    print("\n All processing complete.")