import streamlit as st

import torch

import os
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.segmentation import mark_boundaries

import tempfile
import subprocess

from config import Config
from xai.GradientCAM import GradientCAM
from xai.SaliencyMap import SaliencyMap
from xai.IntegratedGradients import IntegratedGradients
from xai.Lime import Lime
from xai.XAITechniques import XAITechniques

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import Saliency
from lime import lime_image


default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)


def min_max_normalize(heatmap):
    min_value = heatmap.min()
    max_value = heatmap.max()
    normalized_heatmap = (heatmap - min_value) / (max_value - min_value)
    return normalized_heatmap


# Function to make predictions on an image
def make_prediction_image(image, model, test_transforms, device):
    # Apply transforms to image
    frame_tensor = test_transforms(image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(frame_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    predicted_class_name = Config.CLASS_NAMES[predicted_class]
    confidence = round(confidence, 3)

    return predicted_class, predicted_class_name, confidence


def explain_prediction_image(image, model, test_transforms, predicted_class, device):
    # Apply transforms to image
    transformed_image = test_transforms(image)

    # Initialize XAI techniques
    xai_techniques = XAITechniques(
        GradientCAM(GradCAM(model=model, target_layers=[model.resnet50[7][-1]])),
        SaliencyMap(saliency=Saliency(model)), 
        IntegratedGradients(model),
        Lime(explainer=lime_image.LimeImageExplainer(), model=model, device=device)
    )

    # Run XAI techniques
    gradcam, gradcam_visualization = xai_techniques.run_gradcam(transformed_image.unsqueeze(0).to(device), target=predicted_class)
    saliency_map, saliency_map_visualization = xai_techniques.run_saliency_map(transformed_image.unsqueeze(0).to(device), target=predicted_class)
    integrated_gradients = xai_techniques.run_integrated_gradients(transformed_image.unsqueeze(0).to(device), predicted_class)
    lime_heatmap, lime_segments, lime_image_map, lime_mask = xai_techniques.run_lime(transformed_image)

    return gradcam_visualization, saliency_map_visualization, integrated_gradients, lime_heatmap, lime_image_map, lime_mask


def visualize_explanation_heatmaps(gradcam_visualization, saliency_map_visualization, integrated_gradients, lime_heatmap, lime_image_map, lime_mask): 
    # Display explanation heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=600, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

    axes[0, 0].imshow(gradcam_visualization)
    axes[0, 0].set_title('GradCAM', fontsize=22)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(saliency_map_visualization)
    axes[0, 1].set_title('Saliency Map', fontsize=22)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(integrated_gradients, cmap=default_cmap, alpha=0.7)
    axes[0, 2].set_title('Integrated Gradients', fontsize=22)
    axes[0, 2].axis('off')

    axes[1, 0].imshow(mark_boundaries(lime_image_map, lime_mask))
    axes[1, 0].set_title('LIME: Top 3 Superpixels', fontsize=22)
    axes[1, 0].axis('off')

    im = axes[1, 1].imshow(lime_heatmap, cmap='RdBu', vmin=-lime_heatmap.max(), vmax=lime_heatmap.max())
    axes[1, 1].set_title('LIME', fontsize=22)
    cax = fig.add_axes([axes[1, 1].get_position().x1 + 0.01, axes[1, 1].get_position().y0, 0.01, axes[1, 1].get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    axes[1, 1].axis('off')

    # Hide the last subplot as it's empty in a 2x3 layout
    axes[1, 2].axis('off')

    st.pyplot(fig, use_container_width=True)


# Function to format filename
def format_filename(filename):
    # Remove extension, split by underscore, capitalize each word, and join with spaces
    name_parts = os.path.splitext(filename)[0].split('_')
    camel_case_name = ' '.join([part.capitalize() for part in name_parts])
    return camel_case_name

# Function to convert OpenCV frame to PIL Image
def get_pil_image_from_frame(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# Function to convert video to MP4 format using ffmpeg
def convert_to_mp4(video_file):
    # Write video content to a temporary file
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    temp_video_file.write(video_file.read())
    temp_video_file.close()

    # Define output file path for MP4 file
    output_file_path = tempfile.mktemp(suffix='.mp4')

    # Run ffmpeg command for conversion
    command = ['ffmpeg', '-i', temp_video_file.name, '-codec:v', 'libx264', '-codec:a', 'aac', '-strict', 'experimental', output_file_path]
    subprocess.run(command)

    # Clean up temporary video file
    os.remove(temp_video_file.name)

    return output_file_path


# Function to process video
def process_video(video_path, model, test_transforms, device): 
    # Load the video capture
    video_capture = cv2.VideoCapture(video_path)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

    frame_number = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_number += 1

        pil_image = get_pil_image_from_frame(frame)

        # Make prediction on the frame
        predicted_class, confidence = make_prediction_image(pil_image, model, test_transforms, device)
        
        print(predicted_class, confidence)

        if confidence >= Config.CONFIDENCE_THRESHOLD:
            # Display frame with high confidence
            st.image(pil_image, caption=f"Frame {frame_number}: {predicted_class} (Confidence: {confidence})", use_column_width=True)

    # Release video capture
    video_capture.release()
