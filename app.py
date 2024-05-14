""" Import Dependencies """
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model_cnn.hdf5')
    return model

model = load_model()

st.write("""
# Weather Classification Model
""")
st.write("This web app classify weather conditions in uploaded images. Please follow the steps:")
st.markdown("1. Upload an image using the 'Browse Files' button.")
st.markdown("2. Wait for the model to process the image.")
st.markdown("3. View the prediction and confidence score.")
image = Image.open('Weather_girl.jpg')
st.image(image, caption='Weather Classification Model - John Willard S. Sucgang')

with st.container():
    col2 = st.columns((2, 50, 2))
    st.header("Model Outputs")
    st.info("Rain and Shine")

file = st.file_uploader("Choose a photo from your computer", type=["jpg", "png"])

if 'clear_output' not in st.session_state:
    st.session_state.clear_output = False
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
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
        class_names = ['Shine', 'Rain']
        max_prob = np.max(prediction)
        prediction_label = class_names[np.argmax(prediction)]
        # Show the prediction and confidence score
        st.success(f"Prediction: {prediction_label}")
        st.write(f"Confidence Score: {max_prob:.2%}")
        if prediction_label == 'Shine':
            true_labels = ['Shine']  
            predicted_labels = [prediction_label]  
            title = "Confusion Matrix"
        elif prediction_label == 'Rain':
            true_labels = ['Rain'] 
            predicted_labels = [prediction_label]  
            title = "Confusion Matrix"
        else:
            true_labels = [] 
            predicted_labels = []
            title = "Confusion Matrix"
        
        if true_labels and predicted_labels:
            cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
            st.write(title)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

st.info("Github Link: https://github.com/Willythepo0h/Emerging-Tech-2")

comment_tab = st.container()
with comment_tab:
    st.header("User Comments and Feedback")
    st.write("Please leave your comments and feedback about the Weather Classification Model.")
    # Feedback form
    user_name = st.text_input("Your Name:")
    user_email = st.text_input("Your Email:")
    user_comment = st.text_area("Comments:")

    feedback_list = []  # Initialize an empty list to store feedback data

    # Submit button
    if st.button("Submit Feedback"):
        # Store feedback data in a DataFrame or a database
        feedback_data = {
            "Name": user_name,
            "Email": user_email,
            "Comment": user_comment
        }

        # For demonstration purposes, storing feedback in a list
        feedback_list.append(feedback_data)  # Add the current feedback to the list

        # Display a success message to the user
        st.success("Thank you for your feedback! We appreciate your input.")

    # Display feedback history if any feedback is submitted
    if feedback_list:
        st.header("Feedback History")
        feedback_df = pd.DataFrame(feedback_list)  # Create a DataFrame from the feedback list
        st.dataframe(feedback_df)  # Display the feedback data in a DataFrame format
    else:
        st.info("No feedback submitted yet.")
