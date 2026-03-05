import streamlit as st
import requests
import base64
from PIL import Image
import io
import numpy as np

st.set_page_config(layout="wide")

st.title("Flood Water Segmentation")

st.write("Upload a multispectral satellite image (.tif)")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["tif", "tiff"]
)


def decode_image(base64_string):
    img_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_bytes))


if uploaded_file is not None:

    with st.spinner("Running flood detection model..."):

        url = "http://127.0.0.1:5000/predict"

        response = requests.post(
            url,
            files={"image": uploaded_file}
        )

    if response.status_code == 200:

        data = response.json()

        rgb = decode_image(data["image"])
        pred_mask = decode_image(data["pred_mask"])
        overlay = decode_image(data["overlay"])

        rgb = rgb.resize((350, 350))
        pred_mask = pred_mask.resize((350, 350))
        overlay = overlay.resize((350, 350))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("RGB Image")
            st.image(rgb, width=350)

        with col2:

            st.subheader("Predicted Mask")
            st.image(pred_mask, width=350)

            st.divider()

            true_mask_file = st.file_uploader(
                "Drag and drop Ground Truth Mask here",
                type=["png"],
                help="Limit 200MB per file • PNG",
                key="mask_upload"
            )

            if true_mask_file is not None:

                true_mask = Image.open(true_mask_file)

                true_mask = np.array(true_mask)

                true_mask = (true_mask > 0).astype(np.uint8) * 255

                true_mask = Image.fromarray(true_mask).resize((350, 350))

                st.subheader("Ground Truth Mask")

                st.image(true_mask, width=350)

        with col3:
            st.subheader("Overlay")
            st.image(overlay, width=350)

    else:
        st.error("Error from API")