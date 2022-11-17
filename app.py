import time
import spacy
from io import BytesIO
import streamlit as st
from PIL import Image
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel)

st.set_page_config(
    page_title="Handwritten Text Recognition App",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Handwritten Text Recognition")
st.sidebar.subheader("Input")
models_list = ["TrOCR","MEDI-TrOCR"]
network = st.sidebar.selectbox("Select the Model", models_list)

uploaded_file = st.sidebar.file_uploader(
    "Choose an Image for Text Recognition", type=["jpg", "jpeg", "png"]
)
    
if uploaded_file:
    bytes_data = uploaded_file.read()
    inputShape = (224, 224)
    
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
             
    if network in ("TrOCR"):
        @st.cache(allow_output_mutation=True)
        def load_model(model_name):
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
            return (model)
        model = load_model("microsoft/trocr-base-str")
        #model = TrOCRProcessor.from_pretrained("F:/checkpoint-2400/")
    else:
        @st.cache(allow_output_mutation=True)
        def load_model(model_name):
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
            return (model)
        model = load_model("F:/checkpoint-2400/")
        #model = TrOCRProcessor.from_pretrained("F:/checkpoint-2400/")
    image = Image.open(BytesIO(bytes_data))

    image = image.convert("RGB")
    
    def ocr_image(image):             
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    
    ocr_text = ocr_image(image)

    st.image(bytes_data, caption=ocr_text)

    st.success("Handwritten Text Recognition Completed")
