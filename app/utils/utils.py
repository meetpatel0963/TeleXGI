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
