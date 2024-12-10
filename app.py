import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from model.R2GenModel import load_model

# Load the model
MODEL_PATH = 'model/r2gen_model_Bert.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"
# import gzip
# import torch
# import shutil

# compressed_file = "model/r2gen_model_Bert.pth.gz"
# decompressed_file = "model/r2gen_model_Bert.pth"

# with gzip.open(compressed_file, 'rb') as f_in:
#     with open(decompressed_file, 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)
model = load_model(MODEL_PATH)

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit app interface
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Radiologist Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #34495e;'>Upload an X-Ray Image</h4>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=["jpg", "png", "jpeg"],
    help="Limit 200MB per file • JPG, PNG, JPEG"
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess and generate report
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Generate caption
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([[101]]).to(device)  # Start token
            generated_caption = model.generate_caption(input_ids, image_tensor)

        # Clean the output
        report = generated_caption[0] if isinstance(generated_caption, list) else generated_caption
        report = report.replace("generate report :", "").strip()  # Remove prefix and clean up

        # Display the generated report
        st.markdown("<h2 style='color: #2c3e50;'>Report</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #2c3e50; font-size: 16px;'>{report}</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during processing: {e}")
else:
    st.markdown("<p style='text-align: center; color: #bdc3c7;'>No file uploaded yet.</p>", unsafe_allow_html=True)
