import streamlit as st
import tensorflow as tf
import numpy as np
from homePage import cargar_modelo_rnn, cargar_variables_texto

modelo = cargar_modelo_rnn()
vocab, text_vectorization, etiquetas = cargar_variables_texto()

st.header("🧠 Modelos de Redes Neuronales Recurrentes actuando")

st.divider()

user_input_rnn = st.text_input("✍️ Escriba la frase que gustes (en inglés): ")
if user_input_rnn:
    with st.chat_message("user", avatar=":material/emoji_language:"):
        st.write(user_input_rnn)
    st.session_state.user_input = user_input_rnn

user_input = st.session_state.get("user_input", "")
if user_input:
    with st.chat_message("user", avatar=":material/emoji_language:"):
        st.write(user_input)
