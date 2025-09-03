import cv2
import os
import sys
import json
import torch
import numpy as np
import gradio as gr
import safetensors.torch
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2

DRIVE_PATH = "/content/drive/MyDrive/"
PROJECT_PATH = os.path.join(DRIVE_PATH, "SIC_Project_G9/")
CONFIG_FILE_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_PATH, 'config.json')

CFG = None
BEST_MODEL = None
MODEL_NAME = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_config():
    """Load configuration from a JSON file."""
    global CFG
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            CFG = json.load(f)
        return "Configuration loaded successfully."
    except FileNotFoundError:
        return f"Error: {CONFIG_FILE_PATH} not found."
    except json.JSONDecodeError:
        return f"Error: Could not parse {CONFIG_FILE_PATH}."

def apply_clahe_and_median_filter(image_np):
    """Applies Median Filter and CLAHE to a numpy image."""
    image = cv2.medianBlur(image_np, CFG['FILTER_SIZE'])
    lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=CFG['CLIP_LIMIT'], tileGridSize=tuple(CFG['GRID_SIZE']))
    updated_l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((updated_l, a, b)), cv2.COLOR_LAB2RGB)

def add_confidence_heatmap(output, confidence_map):
    """Add confidence heatmap overlay"""
    colored_confidence = cv2.applyColorMap((confidence_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(output, 0.7, colored_confidence, 0.3, 0)

def load_model_from_gdrive():
    """Loads the single .safetensors model from the project folder."""
    global BEST_MODEL, MODEL_NAME
    try:
        safetensors_files = [f for f in os.listdir(PROJECT_PATH) if f.endswith('.safetensors')]
        if not safetensors_files:
            return "Error: No .safetensors model found."
        if len(safetensors_files) > 1:
            return "Error: Multiple .safetensors files found. Please keep only one."

        model_path = os.path.join(PROJECT_PATH, safetensors_files[0])
        MODEL_NAME = '_'.join(safetensors_files[0].split('.')[0].split('_')[2:])
        encoder_name = MODEL_NAME.split('_')[1]

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None)
        state_dict = safetensors.torch.load_file(model_path, device=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        BEST_MODEL = model
        return f"Model '{MODEL_NAME}' loaded on {DEVICE}."
    except Exception as e:
        return f"Error loading model: {e}"

def dice_score(pred_mask, gt_mask):
    """Calculates the Dice score between two binary masks."""
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    if union == 0:
        return 1.0
    return (2.0 * intersection) / union

def iou_score(pred_mask, gt_mask):
    """Calculates the Intersection over Union score."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def predict_and_compare(best_model, mri_image_np, ground_truth_mask_np):
    """Predicts a tumor mask and compares it to a ground truth mask."""
    mri_image_np = np.array(mri_image_np.convert('RGB')) if mri_pil_image else None
    ground_truth_mask_np = np.array(ground_truth_mask_np.convert('L')) if ground_truth_pil_mask else None
    if BEST_MODEL is None:
        return None, None, "Error: Model not loaded. Please check model path."
    if mri_image_np is None:
        return None, None, "Error: Please upload a brain MRI image."

    mri_img_np = cv2.cvtColor(mri_image_np, cv2.COLOR_BGR2RGB).copy()
    processed_image = mri_img_np.copy()
    if not ('raw' in MODEL_NAME.split('_')[0].lower()):
        processed_image = apply_clahe_and_median_filter(processed_image)
    val_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=CFG['NORMALIZE_MEAN'], std=CFG['NORMALIZE_STD']),
        ToTensorV2()])
    augmented = val_transform(image=processed_image)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = BEST_MODEL(input_tensor)
      
    pred_mask = (torch.sigmoid(output).squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    pred_mask_resized = cv2.resize(pred_mask, (mri_image_np.shape[1], mri_image_np.shape[0]))
 
    pred_overlay = mri_img_np.copy()
    contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pred_overlay, contours, -1, (255, 0, 0), 2)  # Red for Prediction

    status_msg = "Prediction successful!"
    comparison_output = None
    if ground_truth_mask_np is not None:
        gt_mask_resized = cv2.resize(ground_truth_mask_np, (mri_image_np.shape[1], mri_image_np.shape[0]))
        dice = dice_score(pred_mask_resized, gt_mask_resized)
        iou = iou_score(pred_mask_resized, gt_mask_resized)
        status_msg += f" | Dice Score: {dice:.4f} | IoU Score: {iou:.4f}"
        
        comparison_output = mri_img_np.copy()
        gt_contours, _ = cv2.findContours(gt_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(comparison_output, gt_contours, -1, (0, 255, 0), 2) # Green for Ground Truth
        cv2.drawContours(comparison_output, contours, -1, (255, 0, 0), 2) # Red for Prediction

    return pred_overlay, comparison_output, status_msg

def clear_interface():
    """Resets the Gradio interface."""
    return None, None, "", gr.update(value=None), gr.update(value=None)

def main():
    try:
        status_text = load_config()
        if not status_text.startswith("Error"):
            status_text = load_model_from_gdrive()
    except Exception as e:
        print(f"Error loading model: {e}")
    
    with gr.Blocks(theme=gr.themes.Soft(), title="🧠👁️") as demo:
        gr.Markdown("# Brain Tumor Segmentation Demo")
        gr.Markdown("Upload a brain MRI scan and an optional ground truth mask. The model will predict the tumor segmentation.")
        with gr.Accordion("📖 How to Use", open=False):
            gr.Markdown("""
            1. Upload a brain MRI scan (required)
            2. Optionally upload ground truth mask for comparison
            3. Click 'Predict' to run segmentation
            4. View results with overlay visualization
            """)
        with gr.Row():
            with gr.Column(scale=1):    
                gr.Markdown("### Upload Images")
                mri_input = gr.Image(type="pil", label="Brain MRI (Required) - 3 Channel RGB")
                gt_mask_input = gr.Image(type="pil", label="Ground Truth Mask (Optional) - Binary Image")
                with gr.Row():
                    predict_button = gr.Button("Predict", variant="primary")
                    clear_button = gr.Button("Clear")
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                with gr.Row():
                    output_overlay = gr.Image(type="numpy", label="Predicted Mask Overlay")
                    comparison_output = gr.Image(type="numpy", label="Comparison (GT vs Pred)")
            
                status_box = gr.Textbox(label="Status", interactive=False, value=status_text)
                metrics_box = gr.Textbox(label="Metrics", interactive=False)
                gr.Markdown(f"**Model Name:** {MODEL_NAME} | **Device:** {DEVICE}")

        predict_button.click(
            fn=predict_and_compare,
            inputs=[mri_input, gt_mask_input],
            outputs=[output_overlay, comparison_output, status_box])

        clear_button.click(
            fn=clear_interface,
            inputs=[],
            outputs=[output_overlay, comparison_output, status_box])
    demo.launch(share=True)

if __name__ == "__main__":
  main()
