import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle

# Configuration
st.set_page_config(page_title="Classification d'Images", layout="centered")
st.title("Classification d'Images")

# Charger le modèle et classes
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model('modele_cnn.h5')
        st.success("✅ Modèle chargé avec succès!")
        return model
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle: {e}")
        return None

@st.cache_resource
def load_class_names():
    try:
        with open('classes.pkl', 'rb') as f:
            class_names = pickle.load(f)
        st.success("✅ Classes chargées avec succès!")
        return class_names
    except Exception as e:
        st.error(f"❌ Erreur chargement classes: {e}")
        return None

# Chargement
model = load_cnn_model()
class_names = load_class_names()

# Section upload - TOUJOURS afficher cette partie
st.header("Uploader une image")
uploaded_file = st.file_uploader(
    "Choisissez une image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Formats supportés: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Afficher l'image
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)
    
    # Classification
    if st.button('Classifier l\'image'):
        with st.spinner('Analyse en cours...'):
            try:
                # Redimensionner selon l'entraînement
                img = image.resize((150, 150))
                img_array = np.array(img) / 255.0
                
                # Adapter les dimensions
                if len(img_array.shape) == 3:  # Image couleur
                    img_array = np.expand_dims(img_array, axis=0)
                
                # Vérification
                st.info(f"Shape de l'image: {img_array.shape}")
                
                # Prédiction
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                # Résultat
                st.success(f"**Classe prédite :** {class_names[predicted_class]}")
                st.info(f"**Confiance :** {confidence:.2%}")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la classification: {e}")
else:
    st.info("📁 Veuillez uploader une image pour commencer la classification")
