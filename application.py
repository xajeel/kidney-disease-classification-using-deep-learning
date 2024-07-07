import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load the saved model
model = load_model(r'trained_model\model.h5')

# Image Preprocessing 
def predict(image_file):
    img = image.load_img(image_file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return np.argmax(prediction, axis=1)[0]


# Streamlit App 

st.title('Kidney Classification Project')
st.divider()

st.header('Upload a CT Scane Image')
st.write('Hello, In this project, we developed a *convolutional neural network (CNN)* model to classify kidney disease using CT scan images. By leveraging deep learning techniques, the model accurately distinguishes between normal and diseased kidneys, aiming to assist in early diagnosis and treatment planning. :sunglasses:')

file_upload = st.file_uploader('Upload File', type=['jpg', 'jpeg', 'png'])
st.divider()

if file_upload is not None:
    img = Image.open(file_upload)
    col1, col2 = st.columns(2)

    with col1:
        resize_image = img.resize((150, 150))
        st.image(resize_image)

    with col2:
        if st.button('Predict'):
            prediction = predict(file_upload)
            if prediction == 0:
                st.success(f'Prediction: Normal')
            else:
                st.success(f'Prediction: Tumor')





