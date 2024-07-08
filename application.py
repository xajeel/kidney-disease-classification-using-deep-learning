import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Load the saved model
model = load_model(r'C:\Users\sajeel\Desktop\kidney-disease-classification-using-deep-learning\trained_model\model2.h5')

# Image Preprocessing 
class_labels = ['Normal', 'Diseased']

def predict(image_path, target_size=(224, 224)):
    target_size=(224, 224)
    img2 = Image.open(image_path)
    img2 = img2.resize(target_size)
    img_array = np.array(img2)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    predicted_class = model.predict(img_array)
    return predicted_class

# Streamlit App 

st.title('Kidney Classification Project')
st.markdown('---')

st.header('Upload a CT Scane Image')
st.write('Hello, In this project, we developed a *convolutional neural network (CNN)* model to classify kidney disease using CT scan images. By leveraging deep learning techniques, the model accurately distinguishes between normal and diseased kidneys, aiming to assist in early diagnosis and treatment planning. :sunglasses:')

file_upload = st.file_uploader('Upload File', type=['jpg', 'jpeg', 'png'])
st.markdown('---')

if file_upload is not None:
    img = Image.open(file_upload)
    col1, col2 = st.columns(2)

    with col1:
        resize_image = img.resize((150, 150))
        st.image(resize_image)

    with col2:
        if st.button('Predict'):
            result = predict(file_upload)
            # st.success(f'Prediction: {result}')
            if result[0][0]  < 0.5:
                st.success(f'Prediction: Normal')
            else:
                st.success(f'Prediction: Tumor')





