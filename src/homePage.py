import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
import numpy as np

def cargar_modelo_tf():
    modelo = keras.models.load_model("data/ModeloTransformers1.h5", compile=False)
    return modelo

def cargar_variables_texto():
    # Cargar vocabulario
    with open("data/vocabulario.txt", "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f.readlines()]

    # Reconstruir TextVectorization
    text_vectorization = TextVectorization(
        max_tokens=20000,
        output_mode="int",
        output_sequence_length=600,
        vocabulary=vocab
    )

    # Cargar etiquetas
    with open("data/etiquetas.txt", "r", encoding="utf-8") as f:
        etiquetas = [line.strip() for line in f.readlines()]

    return vocab, text_vectorization, etiquetas

# * Configuración de los estados de sesión
if "page_icon" not in st.session_state:
    st.session_state.page_icon = "📚"
    
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "rnn" not in st.session_state:
    st.session_state.rnn = False
    
if "transformers" not in st.session_state:
    st.session_state.transformers = False
    
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# * Configuración de la página
st.set_page_config(
    page_title="Clasificador de Emociones", 
    layout="centered", 
    page_icon="📚"
    )
st.title("🧠 Clasificación de Emociones en Texto")
st.write("""Es una tarea de clasificación de texto con el **GoEmotions dataset**:
         \n- El **dataset GoEmotions** contiene 58k comentarios de Reddit cuidadosamente seleccionados y etiquetados según 27 categorías de emociones o Neutral.""")

st.info("""Los modelos neuronales están entrenados con datos en inglés, 
         por lo que, para aprovechar la predicción, 
         es recomendable ingresar frases en inglés.""", icon=":material/info:")

st.write("Para interactuar con el clasificador de emociones, elije uno de los modelos e ingresa una frase y el modelo predecirá la emoción principal y mostrará la confianza de las 5 emociones más probables. ")

st.write("**DATASET**: [Google Research Datasets - GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)")

st.divider()

cols = st.columns(2)

if cols[0].button("RNN", use_container_width=True):
    st.session_state.rnn = True
    st.session_state.transformers = False

# Botón para Transformers
if cols[1].button("Transformers", use_container_width=True):
    st.session_state.rnn = False
    st.session_state.transformers = True

# Mostrar input solo si se ha seleccionado un modelo
if st.session_state.rnn or st.session_state.transformers:
    user_input = st.text_input("✍️ Escriba la frase que gustes (en inglés):")

    if st.button("🔍 Predecir emoción") and user_input:
        st.session_state.user_input = user_input

        if st.session_state.rnn:
            st.switch_page("pages/1_📚_RNN.py")
        elif st.session_state.transformers:
            st.switch_page("pages/2_📚_Transformers.py")


    # texto = tf.convert_to_tensor([[user_input]])
    # vectorizado = text_vectorization(texto)
    # prediccion = modelo(vectorizado).numpy()[0]

    # # Mostrar emoción más probable
    # indice_pred = np.argmax(prediccion)
    # emocion = etiquetas[indice_pred]
    # confianza = prediccion[indice_pred]

    # st.success(f"**Emoción detectada:** {emocion} ({confianza:.2%})")