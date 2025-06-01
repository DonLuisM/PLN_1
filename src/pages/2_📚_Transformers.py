import streamlit as st
from transformers_model import create_summary

st.header("🧠 Modelos Transformers actuando")

st.divider()

user_input = st.text_input("✍️ Escriba la frase que gustes (en inglés): ")
if user_input:
    with st.chat_message("user", avatar=":material/emoji_language:"):
        st.write(user_input)        
        summary = 'Summary.\n' + create_summary(user_input)
        st.write(summary)
