import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model (adjust the path to your keras_model.h5 file)
MODEL_PATH = "C:\\Users\\satis\\OneDrive\\Desktop\\Mangoes\\keras_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess the image for your model
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size to match your model's input
    image = np.array(image)
    image = image.astype('float32') / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title('Mango Ripeness Detector')

uploaded_file = st.file_uploader("Choose a mango image...", type=["jpg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Mango.', use_column_width=True)
        st.write("Classifying...")
        image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(image)
        class_names = ['Raw', 'Ripen', 'In-between']  # Update based on your classes
        st.write(f'Prediction: {class_names[np.argmax(prediction)]}')
    except Exception as e:
        st.error(f"Error: {str(e)}")
