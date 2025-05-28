import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
import numpy as np

if "page_icon" not in st.session_state:
    st.session_state.page_icon = "📚"
    
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "rnn" not in st.session_state:
    st.session_state.rnn = False
    
if "transformers" not in st.session_state:
    st.session_state.transformers = False

st.set_page_config(
    page_title="Clasificador de Emociones", 
    layout="centered", 
    page_icon="📚"
    )
st.title("🧠 Clasificación de Emociones en Texto")
st.write("""Es una tarea de clasificación de texto con el **GoEmotions dataset**: 
         The GoEmotions dataset contiene 58k comentarios de Reddit cuidadosamente seleccionados y etiquetados según 27 categorías de emociones o Neutral. 
         (Los modelos de redes están entrenados con datos en inglés, 
         por lo que, para aprovechar la predicción, 
         es recomendable ingresar frases en inglés.)""")

st.write("Para interactuar con el clasificador de emociones, elije uno de los modelos e ingresa una frase y el modelo predecirá la emoción principal y mostrará la confianza de las 5 emociones más probables. ")

st.divider()

cols = st.columns(2)

if cols[0].button("RNN", use_container_width=True):
    st.session_state.rnn = True
    st.session_state.transformers = False
    
if cols[1].button("Transformers", use_container_width=True):
    st.session_state.rnn = False
    st.session_state.transformers = True
