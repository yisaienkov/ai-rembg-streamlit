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
        st.image(image, width=350)


if __name__ == "__main__":
    session = new_session("u2net_human_seg")
    uploaded_file = st.file_uploader("", type="jpg")

    if uploaded_file is not None:
        input_image = bytes_to_image(uploaded_file)
        output_image = cv2.cvtColor(
            remove(
                cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR),
                session=session, 
            ),
            cv2.COLOR_BGR2RGB,
        )
        
        image_comparison(
            img1=input_image,
            img2=output_image,
        )