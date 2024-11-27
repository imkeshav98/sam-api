import os
import io
import base64
import numpy as np
import torch
from PIL import Image
import json

# Import SAM 2 dependencies
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Initialize Flask app
from flask import Flask, jsonify

app = Flask(__name__)

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load SAM 2 model
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1_hiera_l.yaml"

sam2 = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=32,
    points_per_batch=64,
    pred_iou_thresh=0.56,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    min_mask_region_area=100
)

def generate_masked_image(image_array, masks):
    """
    Generate a masked image with random color overlays for each mask.
    
    Args:
        image_array (np.ndarray): Original image
        masks (list): List of mask dictionaries
    
    Returns:
        np.ndarray: Image with masks overlaid
    """
    # Sort masks by area in descending order
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Create an RGBA image with transparency
    masked_img = image_array.copy().astype(np.float32)
    alpha_overlay = np.zeros((*masked_img.shape[:2], 1), dtype=np.float32)
    
    for mask_info in sorted_masks:
        mask = mask_info['segmentation']
        color = np.random.random(3)  # Random color for each mask
        
        # Create a colored overlay with transparency
        overlay = np.zeros_like(masked_img)
        overlay[mask] = color
        alpha = np.zeros_like(alpha_overlay)
        alpha[mask] = 0.5
        
        # Blend the overlay
        masked_img = np.where(overlay > 0, 
                               overlay * alpha + masked_img * (1 - alpha), 
                               masked_img)
        alpha_overlay = np.maximum(alpha_overlay, alpha)
    
    return (masked_img * 255).astype(np.uint8)

def encode_image(image):
    """
    Encode image to base64 string.
    
    Args:
        image (np.ndarray): Image to encode
    
    Returns:
        str: Base64 encoded image
    """
    pil_img = Image.fromarray(image)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def save_mask_image(mask, output_path):
    """
    Save a single mask as an image.
    
    Args:
        mask (np.ndarray): Mask to save
        output_path (str): Path to save the mask image
    """
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(output_path)

def save_json_data(masks, output_json_path):
    """
    Save the mask details as JSON data.
    
    Args:
        masks (list): List of mask dictionaries
        output_json_path (str): Path to save the JSON data
    """
    mask_data = []
    for mask_info in masks:
        mask_data.append({
            'segmentation': mask_info['segmentation'].tolist(),
            'area': int(mask_info['area']),
            'bbox': [int(x) for x in mask_info['bbox']],
            'predicted_iou': float(mask_info['predicted_iou']),
            'point_coords': mask_info['point_coords'].tolist() if hasattr(mask_info['point_coords'], 'tolist') else mask_info['point_coords'],
            'stability_score': float(mask_info['stability_score']),
            'crop_box': [int(x) for x in mask_info['crop_box']]
        })
    
    with open(output_json_path, 'w') as json_file:
        json.dump(mask_data, json_file, indent=4)

@app.route('/process_images', methods=['GET'])
def process_images():
    """
    Processes all images from the 'input' folder, generates masks, and saves the results.
    
    Returns:
        JSON response with information on processed images and masks.
    """
    input_folder = 'input'  # Folder containing input images
    output_folder = 'output'  # Folder to save output images and masks
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    processed_images = []
    
    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            image_path = os.path.join(input_folder, filename)
            
            try:
                # Open and convert image
                image = Image.open(image_path)
                image_array = np.array(image.convert("RGB"))
                
                # Generate masks
                masks = mask_generator.generate(image_array)
                
                # Generate masked image
                masked_image = generate_masked_image(image_array, masks)
                
                # Save the full masked image
                masked_image_path = os.path.join(output_folder, f"masked_{filename}")
                Image.fromarray(masked_image).save(masked_image_path)
                
                # Save individual mask images and JSON data
                for i, mask_info in enumerate(masks):
                    mask_image_path = os.path.join(output_folder, f"{filename}_mask_{i}.png")
                    save_mask_image(mask_info['segmentation'], mask_image_path)
                
                # Save JSON data
                json_output_path = os.path.join(output_folder, f"{filename}_masks.json")
                save_json_data(masks, json_output_path)
                
                processed_images.append({
                    'filename': filename,
                    'masked_image': masked_image_path,
                    'masks_json': json_output_path,
                    'mask_count': len(masks)
                })
            
            except Exception as e:
                processed_images.append({
                    'filename': filename,
                    'error': str(e)
                })
    
    return jsonify({'processed_images': processed_images})

@app.route('/', methods=['GET'])
def home():
    """Simple home route with usage instructions."""
    return """
    <h1>SAM 2 Mask Generation API</h1>
    <p>Visit the endpoint <code>/process_images</code> to process all images in the 'input_images' folder.</p>
    <p>The results will be saved in the 'output_images' folder, including masked images, individual masks, and JSON data.</p>
    """

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
