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


# # Cargar vocabulario
# with open("data/vocabulario.txt", "r", encoding="utf-8") as f:
#     vocab = [line.strip() for line in f.readlines()]

# # Reconstruir TextVectorization
# text_vectorization = TextVectorization(
#     max_tokens=20000,
#     output_mode="int",
#     output_sequence_length=600,
#     vocabulary=vocab
# )

# # Cargar etiquetas
# with open("data/etiquetas.txt", "r", encoding="utf-8") as f:
#     etiquetas = [line.strip() for line in f.readlines()]

# # Cargar modelo entrenado (sin el vectorizador adentro)
# @st.cache_resource
# def cargar_modelo():
#     modelo = keras.models.load_model("data/ModeloEntrenado.h5", compile=False)
#     return modelo

# modelo = cargar_modelo()

# # Entrada del usuario
# texto_usuario = st.text_input("✏️ Escribe una frase en inglés:")

# if st.button("🔍 Predecir emoción") and texto_usuario:
#     texto = tf.convert_to_tensor([[texto_usuario]])
#     vectorizado = text_vectorization(texto)
#     prediccion = modelo(vectorizado).numpy()[0]

#     # Mostrar emoción más probable
#     indice_pred = np.argmax(prediccion)
#     emocion = etiquetas[indice_pred]
#     confianza = prediccion[indice_pred]

#     st.success(f"**Emoción detectada:** {emocion} ({confianza:.2%})")

#     # Opcional: mostrar top 5
#     top_indices = np.argsort(prediccion)[::-1][:5]
#     st.subheader("Top 5 emociones:")
#     for i in top_indices:
#         st.write(f"- {etiquetas[i]}: {prediccion[i]:.2%}")
