import streamlit as st

st.header("🧠 Modelos Transformers actuando")

st.divider()

user_input = st.text_input("✍️ Escriba la frase que gustes (en inglés): ")
if user_input:
    with st.chat_message("user", avatar=":material/emoji_language:"):
        st.write(user_input)