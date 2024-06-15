import numpy as np
import numpy as np
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import load_model,Model
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import streamlit as st 
import joblib


svc=joblib.load('svm_vgg.pkl')

model = load_model('newVGG.h5')
conv_layer_output = model.get_layer('global_max_pooling2d').output
feature_extraction_model = Model(inputs=model.input, outputs=conv_layer_output)




st.header("Brain Tumor Detector")
input_shape = (224, 224, 3)
photo=Image.open('brain.webp')
st.image(photo)
file=st.file_uploader("Enter the top view of your Brain MRI image", type=['jpg','png','ppm'])



def extract_features(model, img_path):
    img = load_img(img_path, target_size=(224, 224))  # Assuming input size of the model is 224x224
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()


def import_and_predict(image_data,MODEL,svc):
    features = extract_features(feature_extraction_model, image_data)
    value=svc.predict(features.reshape(1, -1))
    if value[0]=='No':
        op=' No tumor'
    else:
        op=" Tumor"
    return op

Button=st.button("Predict")

if file is not None and Button:
    img=Image.open(file)
    new_image = img.resize((200, 200))
    st.image(new_image)
    prediction=import_and_predict(file,model,svc)
    string=f"The uploaded image has high probability that the brain has {prediction}."
    if prediction=='No':
        st.success(string)
    else:
        st.error(string)

elif file is None and Button:
    st.text("please upload an image")