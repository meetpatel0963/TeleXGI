import streamlit as st

import torch

import os
import cv2
from PIL import Image

import tempfile
import subprocess

from config import Config


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

    predicted_class = Config.CLASS_NAMES[predicted_class]
    confidence = round(confidence, 3)

    return predicted_class, confidence


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
