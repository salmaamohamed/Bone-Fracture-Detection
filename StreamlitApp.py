import streamlit as st
import cv2 as cv
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os
import pickle 


IMAGE_SIZE = (256, 256)
K_CLUSTERS = 100 
CLASS_FOLDERS = ['fractured', 'not fractured']

def generate_bone_mask(img):
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    thresh = cv.adaptiveThreshold(
        blurred, 255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY,
        11, 2
    )
    thresh = cv.bitwise_not(thresh)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    return cleaned

def extract_sift_descriptors_single_image(img):
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def create_bow_histogram_single_image(descriptors, kmeans_model, K):
    if descriptors is not None and len(descriptors) > 0:
        descriptors = np.asarray(descriptors, dtype=np.float32)
        print(f"[DEBUG] Descriptor dtype before prediction: {descriptors.dtype}")
        predictions = kmeans_model.predict(descriptors)
        hist, _ = np.histogram(predictions, bins=np.arange(K + 1))
        return hist.reshape(1, -1)
    else:
        return None



import os
@st.cache_resource
def load_models_only():
    try:
        kmeans_path = os.path.abspath("kmeans_model.pkl")
        scaler_path = os.path.abspath("scaler.pkl")
        svm_path = os.path.abspath("svm_model.pkl")

        # st.write(f"🔍 Loading models from:")
        # st.write(f"- KMeans: `{kmeans_path}`")
        # st.write(f"- Scaler: `{scaler_path}`")
        # st.write(f"- SVM: `{svm_path}`")

        with open(kmeans_path, 'rb') as f:
            kmeans_model = pickle.load(f)
            

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        with open(svm_path, 'rb') as f:
            svm_model = pickle.load(f)
            
        return kmeans_model, scaler, svm_model

    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None





def main():
    st.set_page_config(
        page_title="Bone Fracture Detection",
        layout="centered",
        initial_sidebar_state="auto"
    )

    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stFileUploader {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            background-color: #ffffff;
        }
        .stFileUploader label {
            font-weight: bold;
            color: #333;
        }
        .stImage {
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
        }
        .prediction-box {
            background-color: #e6f7ff;
            border-left: 5px solid #2196F3;
            padding: 15px;
            margin-top: 20px;
            border-radius: 5px;
            font-size: 18px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🦴 Bone Fracture Detection")
    st.write("Upload an X-ray image to check for bone fractures.")

    # Load models
    result = load_models_only()

    if result is None:
        st.stop()
    kmeans, scaler, svm = result


    import hashlib
    def model_hash(obj):
        return hashlib.md5(pickle.dumps(obj)).hexdigest()
    
    from joblib import hash as joblib_hash
    # st.write("KMeans hash:", joblib_hash(kmeans))
    # st.write("Scaler hash:", joblib_hash(scaler))
    # st.write("SVM hash:", joblib_hash(svm))

    
    if kmeans is None or scaler is None or svm is None:
        st.error("Models could not be loaded or trained. Please ensure your training script runs correctly to save the models.")
        st.stop()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.write("")
        st.markdown("### Predicting...")

        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)

        if img is None:
            st.error("Error: Could not decode image. Please upload a valid image file.")
            return

        img_resized = cv.resize(img, IMAGE_SIZE)
        mask = generate_bone_mask(img_resized)

        keypoints, descriptors = extract_sift_descriptors_single_image(img_resized)

        if descriptors is None or len(descriptors) == 0:
            st.warning("No SIFT descriptors found for this image. Cannot make a prediction.")
            return

        hist_reshaped = create_bow_histogram_single_image(descriptors, kmeans, K_CLUSTERS)

        if hist_reshaped is None:
            st.error("Could not create Bag-of-Words histogram.")
            return

        try:
            scaled_hist = scaler.transform(hist_reshaped.astype(np.float32))
        except Exception as e:
            st.error(f"Error during scaling: {e}. This might happen if the scaler was not fitted correctly or input dimensions mismatch.")
            return

        try:
            prediction_label = svm.predict(scaled_hist)[0]
            predicted_class = CLASS_FOLDERS[prediction_label]
            print ("text",predicted_class,prediction_label)

            if predicted_class == 'fractured':
                st.markdown(f"<div class='prediction-box' style='background-color: #ffe6e6; border-left: 5px solid #FF4500;'>Prediction: <span style='color: #FF4500;'>**{predicted_class.upper()}**</span></div>", unsafe_allow_html=True)
                st.image("https://placehold.co/300x200/FF4500/FFFFFF?text=FRACTURED", width=200) 
            elif predicted_class == 'not fractured' :
                st.markdown(f"<div class='prediction-box' style='background-color: #e6ffe6; border-left: 5px solid #28A745;'>Prediction: <span style='color: #28A745;'>**{predicted_class.upper()}**</span></div>", unsafe_allow_html=True)
                st.image("https://placehold.co/300x200/28A745/FFFFFF?text=NOT+FRACTURED", width=200) 

        except Exception as e:
            st.error(f"Error during prediction: {e}. Ensure your SVM model is trained and compatible.")
            return

    st.markdown("---")
    st.write("This application uses SIFT features, Bag-of-Words, and an SVM classifier.")
    st.write("For accurate predictions, ensure that the `kmeans_model.pkl`, `scaler.pkl`, and `svm_model.pkl` files (trained with your actual dataset) are available in the same directory as this Streamlit app.")

if __name__ == "__main__":
    import sklearn
    print(sklearn.__version__)  

    main()
    