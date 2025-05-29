import streamlit as st

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
