import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datasets import load_dataset, load_from_disk


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, encoder_outputs, mask=None):
        attention_output_1 = self.attention_1(inputs, inputs)
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(out_1, encoder_outputs)
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)



# Ruta local donde se guardará el dataset
DATASET_PATH = "data/multi_news_local"

# Verificar si el dataset ya está guardado localmente
if os.path.exists(DATASET_PATH):
    print("📁 Cargando dataset desde disco...")
    dataset = load_from_disk(DATASET_PATH)
else:
    print("🌐 Descargando dataset desde Hugging Face...")
    dataset = load_dataset("multi_news", trust_remote_code=True)
    print("💾 Guardando dataset localmente...")
    dataset.save_to_disk(DATASET_PATH)

train_articles = dataset['train']['document']
train_summaries = dataset['train']['summary']

max_target_len = 128
vocab_size = 20000
sequence_length = 600

text_vectorizer = layers.TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)
summary_vectorizer = layers.TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)

text_vectorizer.adapt(train_articles)
summary_vectorizer.adapt(train_summaries)

model = tf.keras.models.load_model("data/transformer_summarizer_model.keras", custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerEncoder": TransformerEncoder,
        "TransformerDecoder": TransformerDecoder
    })

def create_summary(text):
    """
    Generates a summary for a given text using the loaded model.

    Args:
        model: The trained transformer model.
        text: The input text as a string.
        text_vectorizer: The TextVectorization layer used for input text.
        summary_vectorizer: The TextVectorization layer used for target summaries.
        max_target_len: The maximum length of the generated summary.

    Returns:
        The generated summary as a string.
    """
    text_vectorized = text_vectorizer([text])    
    start_token = summary_vectorizer.get_vocabulary()[0]
    decoder_input = np.array([[summary_vectorizer.get_vocabulary().index(start_token)]])

    for i in range(max_target_len):
        predictions = model.predict([text_vectorized, decoder_input])
        predicted_token_id = np.argmax(predictions[:, -1, :], axis=-1)
        decoder_input = np.append(decoder_input, predicted_token_id[:, None], axis=-1)

    generated_summary_ids = decoder_input[0, 1:]
    inverse_vocabulary = summary_vectorizer.get_vocabulary()
    generated_summary = " ".join([inverse_vocabulary[token_id] for token_id in generated_summary_ids if token_id < len(inverse_vocabulary)])

    return generated_summary
