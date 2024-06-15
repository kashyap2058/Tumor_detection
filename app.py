import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img
from tensorflow.keras import models
from PIL import Image, ImageOps
import streamlit as st 


st.header("Brain Tumor Detector")
model=models.load_model("newVGG.h5")
input_shape = (227, 227, 3)
photo=Image.open('brain.webp')
st.image(photo)
file=st.file_uploader("Enter the top view of your Brain MRI image", type=['jpg','png','ppm'])
# Button=st.button("Predict")


def import_and_predict(image_data,MODEL):
    d = {0:'No tumor',1:'Tumor'}
    input_shape = (224, 224, 3)
    test_image = load_img(image_data,target_size = input_shape)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis = 0)
    result = MODEL.predict(test_image)
    value= d[np.argmax(result)]
    return value

Button=st.button("Predict")

if file is not None and Button:
    img=Image.open(file)
    new_image = img.resize((200, 200))
    st.image(new_image)
    prediction=import_and_predict(file,model)
    string=f"The uploaded image has high probability that the brain has {prediction}."
    if prediction=='No tumor':
        st.success(string)
    else:
        st.error(string)

elif file is None and Button:
    st.text("please upload an image")