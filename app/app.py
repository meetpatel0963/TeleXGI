import streamlit as st

import torch
from torchvision import transforms

import os
import time

from PIL import Image

from config import Config
from models.resnet50 import ResNet50Classifier
from utils.utils import make_prediction_image, format_filename


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = ResNet50Classifier().to(device)
checkpoint = torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), Config.RESNET50_MODEL_PATH)), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Defining Transforms
normalize = transforms.Normalize(mean=Config.MEAN, std=Config.STD_DEV)
test_transforms = transforms.Compose([transforms.Resize((Config.RESOLUTION, Config.RESOLUTION)), transforms.ToTensor(), normalize])


# Main Streamlit app
def main():
    st.title('TeleXGI: GastroIntestinal Disease Classification')

    # Upload image or video
    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png'])

    # Example images
    example_images = []
    for filename in os.listdir(Config.SAMPLE_IMAGES_PATH):
        if os.path.isfile(os.path.join(Config.SAMPLE_IMAGES_PATH, filename)):
            example_images.append((format_filename(filename), os.path.join(Config.SAMPLE_IMAGES_PATH, filename)))  # Store tuple of (label, path)

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Convert uploaded image to PIL Image
        pil_image = Image.open(uploaded_file)

        # Make prediction on the image
        predicted_class, confidence = make_prediction_image(pil_image, model, test_transforms, device)
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Confidence: {confidence}")

    selected_example_media = st.sidebar.selectbox("Select an example image", example_images, format_func=lambda x: x[0])

    # Display example image
    example_image = Image.open(selected_example_media[1])
    st.sidebar.image(example_image, caption='Example Image', use_column_width=True)

    # Make prediction on example image
    predicted_class, confidence = make_prediction_image(example_image, model, test_transforms, device)
    st.sidebar.write(f"Predicted class: {predicted_class}")
    st.sidebar.write(f"Confidence: {confidence}")


if __name__ == "__main__":
    main()

