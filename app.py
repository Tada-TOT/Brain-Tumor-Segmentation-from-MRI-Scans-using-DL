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
CONFIG_FILE_PATH = CONFIG_FILE_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_PATH, 'config.json')
CFG = None
MODEL_NAME = None
try:
    with open(CONFIG_FILE_PATH, 'r') as f:
        CFG = json.load(f)
except FileNotFoundError:
    print(f"Error: {CONFIG_FILE_PATH} not found.")

def apply_clahe_and_median_filter(image_np):
    """Applies Median Filter and CLAHE to a numpy image."""
    image = cv2.medianBlur(image_np, CFG['FILTER_SIZE'])
    lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=CFG.CLIP_LIMIT, tileGridSize=CFG.GRID_SIZE)
    updated_l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((updated_l, a, b)), cv2.COLOR_LAB2RGB)

def load_model_from_gdrive():
    """
    Dynamically finds the single .safetensors file in the project folder and loads the model.
    """
    device = torch.device("cpu")
    safetensors_files = [f for f in os.listdir(PROJECT_PATH) if f.endswith('.safetensors')]
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors file found in {PROJECT_PATH}. Please check your model saving process.")
    if len(safetensors_files) > 1:
        raise ValueError(f"Multiple .safetensors files found in {PROJECT_PATH}. Please ensure only one exists.")
    model_path = os.path.join(PROJECT_PATH, safetensors_files[0])
    print(f"Found and loading model from: {model_path}")

    MODEL_NAME = safetensors_file.split('.')[0]
    encode_name = MODEL_NAME.split('_')[1]
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None)
    
    state_dict = safetensors.torch.load_file(model_path, device=str(DEVICE))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def predict_and_compare(mri_image_np, ground_truth_mask_np):
    if best_model is None:
        return None, None, "Model not loaded. Please check the path and try again."    
    if mri_image_np is None:
        return None, None, "Error: Please upload a brain MRI image."

    processed_image = mri_image_np.copy()
    if not ('raw' in MODEL_NAME.split('_')[0].lower()):
        processed_image = apply_clahe_and_median_filter(processed_image)
    val_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=CFG['NORMALIZE_MEAN'], std=CFG['NORMALIZE_STD']),
        ToTensorV2()])
    augmented = val_transform(image=processed_image)
    input_tensor = augmented['image'].unsqueeze(0)
    
    with torch.no_grad():
        output = best_model(input_tensor)
      
    pred_mask = (torch.sigmoid(output).squeeze().numpy() > 0.5).astype(np.uint8) * 255
    
    overlay = mri_image_np.copy()
    pred_mask_resized = cv2.resize(pred_mask, (mri_image_np.shape[1], mri_image_np.shape[0]))
    contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    comparison_output = None
    if ground_truth_mask_np is not None and ground_truth_mask_np.sum() > 0:
        gt_mask_resized = cv2.resize(ground_truth_mask_np, (mri_image_np.shape[1], mri_image_np.shape[0]))
        
        comparison_output = mri_image_np.copy()
        gt_contours, _ = cv2.findContours(gt_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(comparison_overlay, gt_contours, -1, (0, 255, 0), 2) # Green for Ground Truth
        cv2.drawContours(comparison_overlay, pred_contours, -1, (255, 0, 0), 2) # Red for Prediction

    return overlay, comparison_output, "Prediction successful!"

def main():
  try:
      best_model = load_model_from_gdrive()
      print("Model loaded successfully!")
  except Exception as e:
      print(f"Error loading model: {e}")
      best_model = None
    
  with gr.Blocks(title="Brain Tumor Segmentation Demo") as demo:
      gr.Markdown("# Brain Tumor Segmentation Demo")
      gr.Markdown("Upload a brain MRI scan and an optional ground truth mask to see the predicted tumor segmentation.")
      
      with gr.Row():
          with gr.Column():
              mri_input = gr.Image(type="numpy", label="Upload Brain MRI (Required)")
              gt_mask_input = gr.Image(type="numpy", label="Upload Ground Truth Mask (Optional)")
              run_button = gr.Button("Predict")
          with gr.Column():
              output_overlay = gr.Image(type="numpy", label="Predicted Mask Overlay")
              comparison_output = gr.Image(type="numpy", label="Comparison (GT vs Pred)")
              status_text = gr.Textbox(label="Status")
      run_button.click(
          fn=predict_and_compare,
          inputs=[mri_input, gt_mask_input],
          outputs=[output_overlay, comparison_output, status_text])
      demo.launch(share=True)

if __name__ == "__main__":
  main()
