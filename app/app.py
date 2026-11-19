import streamlit as st 
import sys 
from pathlib import Path
from PIL import Image

ROOT= Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict import load_class_names, load_model, predict_pil_image

st.set_page_config(
    page_title="Pet predict",
    page_icon="🐾",
    layout="centered"
)
@st.cache_resource
def get_model_and_labels():
    return load_model(), load_class_names()

model, class_names= get_model_and_labels()

st.title("🐾 Pet Predict")
st.write("Upload a photograph of a cat or a dog, The model will recoganize it!")

uploaded_file= st.file_uploader(
    "Choose an image",
    type=["jpg","jpage","png"]
)

if uploaded_file is None:
    st.info("No image yet. Upload one!")
    st.stop()
image = Image.open(uploaded_file).convert("RGB")

left, right= st.columns([1,1])

with left:
    st.image(image, caption="Upleaded iamge", use_container_width=True)

with right:
    with st.spinner("Predicting..."):
        label, Confidence,scores= predict_pil_image(model,class_names,image
        )

    st.success(f"Prediction:**{label}**")
    st.metric("Confidence",f"{Confidence:.2%}")

    st.write("Scores dor each pet:")
    for name, score in zip(class_names,scores):
        st.write(f"**{name}**-{score:2%}")
        st.progress(float(score))

st.caption(
    "This model is trained on a smale dataset. "
    "It may be wrong on ususual photos. "
    "It can only predict cat or dog even if you use other animal photo!"
)
