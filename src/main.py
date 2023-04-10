import io
import math
import zipfile

import cv2
import numpy as np
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from rembg import remove, new_session
from streamlit_image_comparison import image_comparison


def bytes_to_image(file: bytes) -> np.array:
    bytes_data = file.getvalue()
    np_array = np.fromstring(bytes_data, np.uint8)

    image_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    return image_array


def vis_image(image: np.array, col: DeltaGenerator) -> None:
    with col:
        st.image(image)


def vis_image_comparison(first_file, session):
    input_image = bytes_to_image(first_file)
    output_image_bgr = remove(
        cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR),
        session=session, 
    )
    output_image_rgb = cv2.cvtColor(
        output_image_bgr,
        cv2.COLOR_BGR2RGB,
    )
    
    image_comparison(
        img1=input_image,
        img2=output_image_rgb,
        in_memory=True,
        show_labels=False,
    )


def vis_grid(uploaded_files, session, n_cols=4):
    memory_file = io.BytesIO()
    rows = [st.columns(n_cols) for _ in range(math.ceil(len(uploaded_files) / n_cols))]
    
    with zipfile.ZipFile(memory_file, mode='w') as zip_file:
        for i, uploaded_file in enumerate(uploaded_files):
            input_image = bytes_to_image(uploaded_file)
            output_image_bgr = remove(
                cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR),
                session=session, 
            )
            output_image_rgb = cv2.cvtColor(
                output_image_bgr,
                cv2.COLOR_BGR2RGB,
            )
            vis_image(output_image_rgb, rows[i // n_cols][i % n_cols])

            _, buffer = cv2.imencode('.png', output_image_bgr)
            zip_file.writestr(f'image_{i}.png', buffer)
        
    memory_file.seek(0)

    st.download_button("Download Images", memory_file, file_name="images.zip")


if __name__ == "__main__":
    session = new_session("u2net_human_seg")
    uploaded_files = st.file_uploader("", type="jpg", accept_multiple_files=True)

    if uploaded_files:
        vis_image_comparison(uploaded_files[0], session)

        vis_grid(uploaded_files, session)

