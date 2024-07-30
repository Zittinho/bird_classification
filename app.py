import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
import streamlit as st
import json


model_save_path = 'C:\\Users\\Motka\\Desktop\\bird_classification\\models\\bird_classification_model.keras'
class_names_path = 'C:\\Users\\Motka\\Desktop\\bird_classification\\models\\class_names.json'


if os.path.exists(model_save_path):
    loaded_model = tf.keras.models.load_model(model_save_path)
else:
    loaded_model = None


if os.path.exists(class_names_path):
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
else:
    class_names = []


st.title("Bird Classification App")
st.write("Загрузите изображение птицы, чтобы классифицировать её.")


uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if loaded_model is not None and class_names:
        
        predictions = loaded_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_class_name = class_names[predicted_class[0]]
        
        
        st.image(img, caption='Загруженное изображение.', use_column_width=True)
        st.write(f"Предсказанный класс: {predicted_class_name}")
    else:
        st.write("Модель не загружена или классы не определены. Пожалуйста, обучите модель перед использованием.")
