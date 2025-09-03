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
try:
    with open(CONFIG_FILE_PATH, 'r') as f:
        CFG = json.load(f)
except FileNotFoundError:
    print(f"Error: {CONFIG_FILE_PATH} not found.")

def load_model_from_gdrive():
    """
    Dynamically finds the single .safetensors file in the project folder and loads the model.
    """
    device = torch.device("cpu")
    safetensors_file = [f for f in os.listdir(PROJECT_PATH) if f.endswith('.safetensors')][0]
    model_path = os.path.join(PROJECT_PATH, safetensors_file)
    print(f"Found and loading model from: {model_path}")

    encode_name = safetensors_file.split('_')[1].split('.')[0]
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_and_compare(mri_image_np, ground_truth_mask_np):
    if best_model is None:
        return None, None, "Model not loaded. Please check the path and try again."

    # Preprocessing must be IDENTICAL to validation/test preprocessing
    val_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=CFG['NORMALIZE_MEAN'], std=CFG['NORMALIZE_STD']),
        ToTensorV2()])
  
    # Preprocess the MRI image
    augmented = val_transform(image=mri_image_np)
    input_tensor = augmented['image'].unsqueeze(0)
    
    with torch.no_grad():
        output = best_model(input_tensor)
      
    pred_mask = (torch.sigmoid(output).squeeze().numpy() > 0.5).astype(np.uint8) * 255
    
    overlay = mri_image_np.copy()
    pred_mask_resized = cv2.resize(pred_mask, (mri_image_np.shape[1], mri_image_np.shape[0]))
    contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    comparison_overlay = None
    if ground_truth_mask_np is not None and ground_truth_mask_np.sum() > 0:
        gt_mask_resized = cv2.resize(ground_truth_mask_np, (mri_image_np.shape[1], mri_image_np.shape[0]))
        
        comparison_overlay = mri_image_np.copy()
        gt_contours, _ = cv2.findContours(gt_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(comparison_overlay, gt_contours, -1, (0, 255, 0), 2) # Green for Ground Truth
        cv2.drawContours(comparison_overlay, pred_contours, -1, (255, 0, 0), 2) # Red for Prediction
    
    pred_mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)

    return overlay, comparison_overlay, "Prediction successful!"

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
