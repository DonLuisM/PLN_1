import streamlit as st
import tensorflow as tf
import numpy as np
from homePage import cargar_modelo_rnn, cargar_variables_texto

modelo = cargar_modelo_rnn()
vocab, text_vectorization, etiquetas = cargar_variables_texto()

st.header("🧠 Modelos de Redes Neuronales Recurrentes actuando")

st.divider()

user_input_rnn = st.text_input("✍️ Escriba la frase que gustes (en inglés): ", placeholder="You are beautiful")

st.caption("EX1. Thank you. I really appreciate your response")
st.caption("EX2. Haha, love it! :)")
st.caption("EX3. Good luck my G")

submit_phrase = st.button("Predecir emoción", icon="🔍")

if user_input_rnn:
    if submit_phrase:
        with st.chat_message("user", avatar=":material/emoji_language:"):
            st.write(user_input_rnn)
        st.session_state.user_input = user_input_rnn
        
        texto = tf.convert_to_tensor([[st.session_state.user_input]])
        vectorizado = text_vectorization(texto)
        prediccion = modelo(vectorizado).numpy()[0]

        # Mostrar emoción más probable
        indice_pred = np.argmax(prediccion)
        emocion = etiquetas[indice_pred]
        confianza = prediccion[indice_pred]

        st.success(f"**Emoción detectada:** {emocion} ({confianza:.2%})")

        # Mostrar top de emociones de clasificación
        top_indices = np.argsort(prediccion)[::-1][:5]
        st.subheader("Top 5 emociones:")
        for i in top_indices:
            st.write(f"- {etiquetas[i]}: {prediccion[i]:.2%}")

