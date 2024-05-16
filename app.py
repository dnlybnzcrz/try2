""" Import Dependencies """
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_cnn.hdf5')
    return model

model = load_model()

st.write("""
# Weather Classification Model
""")
st.write("This web app classifies weather conditions in uploaded images. Please follow the steps:")
st.markdown("1. Upload an image using the 'Browse Files' button.")
st.markdown("2. Wait for the model to process the image.")
st.markdown("3. View the prediction and confidence score.")
image = Image.open('Weather_girl.jpg')
st.image(image, caption='Weather Classification Model - John Willard S. Sucgang')

st.header("Model Outputs")
st.info("Rain and Shine")

file = st.file_uploader("Choose a photo from your computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (224, 224)  # Ensure this matches your model input size
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    if file.type not in ['image/jpeg', 'image/png']:
        st.error("Unsupported file type. Please upload a .jpg or .png file.")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        # Process the image and get predictions
        prediction = import_and_predict(image, model)
        class_names = ['Shine', 'Rain']  # Adjust to your actual class names
        max_prob = np.max(prediction)
        prediction_label = class_names[np.argmax(prediction)]
        # Show the prediction and confidence score
        st.success(f"Prediction: {prediction_label}")
        st.write(f"Confidence Score: {max_prob:.2%}")

        # For demonstration, we'll create a mock confusion matrix
        true_labels = ['Shine', 'Rain', 'Shine', 'Rain']  # Mock true labels
        predicted_labels = [prediction_label, prediction_label, 'Shine', 'Rain']  # Mock predictions
        
        cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
        st.write("Confusion Matrix")
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        st.pyplot(plt)

st.info("Github Link: https://github.com/Willythepo0h/Emerging-Tech-2")

st.header("User Comments and Feedback")
st.write("Please leave your comments and feedback about the Weather Classification Model.")

if 'feedback_list' not in st.session_state:
    st.session_state.feedback_list = []

user_name = st.text_input("Your Name:")
user_email = st.text_input("Your Email:")
user_comment = st.text_area("Comments:")

if st.button("Submit Feedback"):
    feedback_data = {
        "Name": user_name,
        "Email": user_email,
        "Comment": user_comment
    }
    st.session_state.feedback_list.append(feedback_data)
    st.success("Thank you for your feedback! We appreciate your input.")

if st.session_state.feedback_list:
    st.header("Feedback History")
    feedback_df = pd.DataFrame(st.session_state.feedback_list)
    st.dataframe(feedback_df)
else:
    st.info("No feedback submitted yet.")
