import streamlit as st
import os
from PIL import Image
import numpy as np
from tensorflow.keras.layers import GlobalMaxPooling2D  
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import pickle
from numpy.linalg import norm

# Load embeddings and filenames
feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load pre-trained ResNet50 model
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

# Function to save uploaded file
def save_uploadedfile(uploadedfile):
    try:
        save_path = os.path.join("D:/Programming/Project/Fashion Recommendation System/upload", uploadedfile.name)
        with open(save_path, "wb") as f:
            f.write(uploadedfile.getbuffer())
        return save_path  # Return the saved file path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Function to extract features from an image
def feature_extract(img_path, model):
    img = load_img(img_path, target_size=(224, 224))  # Load image
    img_array = img_to_array(img)  # Convert to numpy array
    expanded_img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    preprocess_img = preprocess_input(expanded_img_array)  # Preprocess image
    result = model.predict(preprocess_img).flatten()  # Get predictions
    normalized_result = result / norm(result)  # Normalize features
    return normalized_result

def recommend(features,feature_list):
   neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
   neighbors.fit(feature_list)

   distances,indices = neighbors.kneighbors([features])
   
   return indices 
# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    saved_file_path = save_uploadedfile(uploaded_file)  # Save the file

    if saved_file_path:
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_container_width=True)

        # Extract features from the uploaded image
        features = feature_extract(saved_file_path, model)
        st.text(f"Extracted Features: {features}")
        
        indices = recommend(features,feature_list)
        
        col1,col2,col3,col4,col5 = st.columns(5)
        
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.error("Some error occurred while uploading the file.")
