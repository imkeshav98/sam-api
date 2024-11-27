import os
import io
import base64
import numpy as np
import torch
from PIL import Image
import flask
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import SAM 2 dependencies
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Initialize Flask app
app = Flask(__name__)

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load SAM 2 model
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "./configs/sam2.1_hiera_l.yaml"

sam2 = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    min_mask_region_area=25.0
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

def encode_segmentation_mask(mask):
    """
    Encode segmentation mask to base64 string.
    
    Args:
        mask (np.ndarray): Binary segmentation mask
    
    Returns:
        str: Base64 encoded mask
    """
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    buffered = io.BytesIO()
    mask_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/generate_masks', methods=['POST'])
def generate_masks():
    """
    API endpoint for generating masks from an uploaded image.
    
    Returns:
        JSON response with masked image and comprehensive mask details
    """
    # Check if image is present
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Read the image
    file = request.files['image']
    
    # Validate the image file format
    if not file.filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
        return jsonify({"error": "Invalid image file format"}), 400
    
    # Save the file temporarily
    temp_dir = '/tmp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(temp_dir, filename)
    file.save(file_path)
    
    try:
        # Open and convert image
        image = Image.open(file_path)
        image_array = np.array(image.convert("RGB"))
        
        # Generate masks
        try:
            masks = mask_generator.generate(image_array)
        except Exception as e:
            return jsonify({"error": f"Mask generation failed: {str(e)}"}), 500
        
        # Generate masked image
        masked_image = generate_masked_image(image_array, masks)
        
        # Prepare comprehensive mask data for JSON response
        mask_data = []
        for mask_info in masks:
            mask_data.append({
                'segmentation': encode_segmentation_mask(mask_info['segmentation']),
                'area': int(mask_info['area']),
                'bbox': [int(x) for x in mask_info['bbox']],
                'predicted_iou': float(mask_info['predicted_iou']),
                'point_coords': mask_info['point_coords'].tolist() if hasattr(mask_info['point_coords'], 'tolist') else mask_info['point_coords'],
                'stability_score': float(mask_info['stability_score']),
                'crop_box': [int(x) for x in mask_info['crop_box']]
            })
        
        # Encode images
        original_b64 = encode_image(image_array)
        masked_b64 = encode_image(masked_image)
        
        # Return response
        return jsonify({
            'original_image': original_b64,
            'masked_image': masked_b64,
            'mask_count': len(masks),
            'masks': mask_data
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)

@app.route('/', methods=['GET'])
def home():
    """Simple home route with usage instructions."""
    return """
    <h1>SAM 2 Mask Generation API</h1>
    <p>Send a POST request to /generate_masks with an image file.</p>
    <p>Use form-data with key 'image' and the image file as the value.</p>
    """

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
