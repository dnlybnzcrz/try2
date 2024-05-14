import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model_cnn.hdf5')
    return model

def test_title():
    expected_title = "Weather Classification Model"

    # Use st.title to display the title
    current_title = st.title

    return current_title == expected_title

if __name__ == '__main__':
    if test_title():
        print("Title is correct")
    else:
        print("Title has changed")

def test_model_hash():
    # Implement a method to check the hash of the model file and compare it with the previous hash
    # If the hash has changed, return True; otherwise, return False
    pass

if __name__ == '__main__':
    if test_title():
        print("Title is correct")
    else:
        print("Title has changed")

    if test_model_hash():
        print("Model has changed")
    else:
        print("Model is the same")
