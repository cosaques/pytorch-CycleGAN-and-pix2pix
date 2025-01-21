import streamlit as st
import requests
from PIL import Image
import io
import os

dir_path = os.path.dirname(__file__)

# Set endpoint based on environment
API_URL = "https://pix2pix-193679124808.europe-west1.run.app/process_tshirt/"

# Page title and description
st.title("T-shirt Image Transformation")
st.write(
    """
    Upload or take a picture of your t-shirt, and our AI will transform it into a professional-looking product photo.
    The first run might take some time, but subsequent ones will be faster as the AI model is loaded into memory.
    Here is an example:
    """
)

# Display table
col1, col2, col3 = st.columns(3)
# Add text to each column without borders
with col1:
    st.markdown("<p style='text-align: center; font-size: 20px; font-weight: bold;'>Your picture</p>", unsafe_allow_html=True)
with col2:
    st.markdown("<p style='text-align: center; font-size: 20px; font-weight: bold;'>AI generated photo</p>", unsafe_allow_html=True)
with col3:
    st.markdown("<p style='text-align: center; font-size: 20px; font-weight: bold;'>Real professional photo</p>", unsafe_allow_html=True)

# Display process image
process_image_path = os.path.join(dir_path, "src", "t-shirts-3.jpeg")
st.image(process_image_path, caption="Transformation Process", use_container_width=True)

# Upload image section
uploaded_file = st.file_uploader("Upload your t-shirt image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Prepare the file for sending
    uploaded_file.seek(0)  # Reset the file pointer to the beginning
    file_bytes = uploaded_file.read()  # Read file content as bytes
    files = {"image": (uploaded_file.name, file_bytes, uploaded_file.type)}

    # Display the spinner while making the HTTP request
    with st.spinner("Your t-shirt is being transformed, please wait..."):
        try:
            # Send POST request to API
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                # Read the image directly from response content
                result_image = Image.open(io.BytesIO(response.content))
                st.image(result_image, caption="Transformed T-shirt", use_container_width=True)
            else:
                st.error("T-shirt not detected in the image, please try again.")
        except requests.exceptions.RequestException as e:
            st.error("An error occurred while processing your image. Please try again.")
