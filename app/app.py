import streamlit as st

import torch
from torchvision import transforms

import os
import time

from PIL import Image

from config import Config
from models.resnet50 import ResNet50Classifier
from utils.utils import make_prediction_image, get_pil_image_from_frame, \
                        format_filename, convert_to_mp4, process_video


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
    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])

    # Example images
    example_images = []
    for filename in os.listdir(Config.SAMPLE_IMAGES_PATH):
        if os.path.isfile(os.path.join(Config.SAMPLE_IMAGES_PATH, filename)):
            example_images.append((format_filename(filename), os.path.join(Config.SAMPLE_IMAGES_PATH, filename)))  # Store tuple of (label, path)

    # Example videos
    example_videos = []
    for filename in os.listdir(Config.SAMPLE_VIDEOS_PATH):
        if os.path.isfile(os.path.join(Config.SAMPLE_VIDEOS_PATH, filename)):
            example_videos.append((format_filename(filename), os.path.join(Config.SAMPLE_VIDEOS_PATH, filename)))  # Store tuple of (label, path)

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ['mp4', 'avi']:
            if file_extension == 'avi':
                # Convert uploaded video to MP4 format
                # video_path = convert_to_mp4(uploaded_file)
                video_path = uploaded_file

                # Process video and display frames with high confidence
                process_video(video_path, model, test_transforms, device)
        else:
            # Display uploaded image
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

            # Convert uploaded image to PIL Image
            pil_image = Image.open(uploaded_file)

            # Make prediction on the image
            predicted_class, confidence = make_prediction_image(pil_image, model, test_transforms, device)
            st.write(f"Predicted class: {predicted_class}")
            st.write(f"Confidence: {confidence}")

    st.sidebar.title("Example Media")
    selected_media_type = st.sidebar.radio("Select media type", ["Image", "Video"])

    if selected_media_type == "Image":
        selected_example_media = st.sidebar.selectbox("Select an example image", example_images, format_func=lambda x: x[0])

        # Display example image
        example_image = Image.open(selected_example_media[1])
        st.sidebar.image(example_image, caption='Example Image', use_column_width=True)

        # Make prediction on example image
        predicted_class, confidence = make_prediction_image(example_image, model, test_transforms, device)
        st.sidebar.write(f"Predicted class: {predicted_class}")
        st.sidebar.write(f"Confidence: {confidence}")

    elif selected_media_type == "Video":
        selected_example_media = st.sidebar.selectbox("Select an example video", example_videos, format_func=lambda x: x[0])

        # Convert uploaded video to MP4 format
        # video_path = convert_to_mp4(open(selected_example_media[1], 'rb'))
        video_path = selected_example_media[1]

        # Display converted video
        # st.video(open(video_path, 'rb'), format='video/mp4')

        # Process video and display frames with high confidence
        process_video(video_path, model, test_transforms, device)


if __name__ == "__main__":
    main()

