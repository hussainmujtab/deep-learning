import tensorflow
import transformers
from tensorflow import keras
from keras.applications import InceptionV3, Xception, VGG19, VGG16, ResNet50
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.utils import img_to_array
from keras.utils import load_img
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel)
from transformers import TrOCRPreTrainedModel
from io import BytesIO
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import urllib
import cv2

headers = {
    st.write("DB username:", st.secrets["db_username"]),
    st.write("DB password:", st.secrets["db_password"]),
    st.write("My cool secrets:", st.secrets["my_cool_secrets"]["things_i_like"])
}

st.set_page_config(
    page_title="Handwritten Text Recognition App",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.subheader("Input")
models_list = ["TrOCR","MEDI-TrOCR", "VGG16", "VGG19", "Inception", "Xception", "ResNet"]
network = st.sidebar.selectbox("Select the Model", models_list)


MODELS = {
    "TrOCR": TrOCRPreTrainedModel,
    "MEDI-TrOCR": TrOCRPreTrainedModel,
    "VGG16": VGG16,
    "VGG19": VGG19,
    "Inception": InceptionV3,
    "Xception": Xception,
    "ResNet": ResNet50,
}

uploaded_file = st.sidebar.file_uploader(
    "Choose an Image for Text Recognition", type=["jpg", "jpeg", "png"]
)
    
if uploaded_file:
    bytes_data = uploaded_file.read()
    inputShape = (224, 224)
    
                 
    if network == ("TrOCR"):
        st.title("Handwritten Text Recognition")
        @st.cache(allow_output_mutation=True)
        def load_model(model_name):
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
            return (model)
        model = load_model("microsoft/trocr-base-str")
        image = Image.open(BytesIO(bytes_data))

        image = image.convert("RGB")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
        def ocr_image(image):             
            pixel_values = processor(image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text
    
        ocr_text = ocr_image(image)

        st.image(bytes_data, caption=ocr_text)
        st.success("Handwritten Text Recognition Completed")
        #model = TrOCRProcessor.from_pretrained("F:/checkpoint-2400/")
    elif network == ("MEDI-TrOCR"):
        st.title("Handwritten Text Recognition")
        @st.cache(allow_output_mutation=True)
        def load_model(model_name):
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
            return (model)
        model = load_model("F:/checkpoint-2400/")
        #model = TrOCRProcessor.from_pretrained("F:/checkpoint-2400/")
        image = Image.open(BytesIO(bytes_data))

        image = image.convert("RGB")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
        def ocr_image(image):             
            pixel_values = processor(image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text
    
        ocr_text = ocr_image(image)

        st.image(bytes_data, caption=ocr_text)
        st.success("Handwritten Text Recognition Completed")
    elif network in ("VGG16", "VGG19", "ResNet"):
        st.title(f"Image Classification from {network} Model")
        preprocess = imagenet_utils.preprocess_input
        Network = MODELS[network]
        model = Network(weights="imagenet")
        image = Image.open(BytesIO(bytes_data))
        image = image.convert("RGB")
        image = image.resize(inputShape)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess(image)
        preds = model.predict(image)
        predictions = imagenet_utils.decode_predictions(preds)
        imagenetID, label, prob = predictions[0][0]
        st.image(bytes_data, caption=[f"{label} {prob*100:.2f}"])
        st.subheader(f"Top Predictions from {network}")
        st.dataframe(
            pd.DataFrame(
                predictions[0], columns=["Network", "Classification", "Confidence"]
            )
        )
    elif network in ("Inception", "Xception"):
        st.title(f"Image Classification from {network} Model")
        inputShape = (299, 299)
        preprocess = imagenet_utils.preprocess_input
        Network = MODELS[network]
        model = Network(weights="imagenet")
        image = Image.open(BytesIO(bytes_data))
        image = image.convert("RGB")
        image = image.resize(inputShape)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess(image)
        preds = model.predict(image)
        predictions = imagenet_utils.decode_predictions(preds)
        imagenetID, label, prob = predictions[0][0]
        st.image(bytes_data, caption=[f"{label} {prob*100:.2f}"])
        st.subheader(f"Top Predictions from {network}")
        st.dataframe(
            pd.DataFrame(
                predictions[0], columns=["Network", "Classification", "Confidence"]
            )
        )
