# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 21:41:32 2021

@author: my
"""

from google.colab import drive
drive.mount('/content/drive')

from transformers import pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from dataclasses import dataclass
import pandas as pd
import numpy as np
import glob
import re
from pprint import pprint
from transformers import BertTokenizer

data = pd.read_csv('/content/drive/MyDrive/finaltoeic2.csv')

data=data.drop([407,4254,4344,598]).reset_index()

class Config:
    MAX_LEN = 46
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 30522
    EMBED_DIM = 128
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1


config = Config()

from keras.preprocessing.sequence import pad_sequences
bertmodel = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bertmodel)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

MAX_LEN=46
tokenized_texts = [tokenizer.tokenize(sent) for sent in data.sentence]
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")        
input_ids

encoded_texts=input_ids
inp_mask = encoded_texts ==  103

# label도 input과 같은 길이의 배열로 만들어 준후 
# 위에서 만든 inp_mask를 넣어서 MASK 부분만 3 ,나머지는 -1로 바꾼다
labels = -1 * np.ones(encoded_texts.shape, dtype=int)
labels[inp_mask] = encoded_texts[inp_mask]

input_ids = np.copy(encoded_texts)

# Prepare sample_weights to pass to .fit() method
sample_weights = np.ones(labels.shape)
sample_weights[labels == -1] = 0

# y_labels would be same as encoded_texts i.e input tokens
# y_labels=[]
# for j in range(len(data)):
#   A=""
#   new_sentence= data.sentence[j].split(" ")
#   for i in range(len(new_sentence)):
#     if  new_sentence[i] == "[MASK]":
#       new_sentence[i] = data.label[j]
#     A+=" "+new_sentence[i]
#   y_labels.append(A) 
   
# label_ids=encode(y_labels)




MAX_LEN=4
tokenized_texts = [tokenizer.tokenize(sent) for sent in data.label]
label_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

one_hot_vectors=[]

for i in range(len(label_ids)):
  one_hot_vector = [0]*(int(30522))
  for j in label_ids[i]:
      j= int(j)
      if j != 0 : 
        one_hot_vector.insert(j,1)
  one_hot_vectors.append(one_hot_vector)
  
label_ids = pad_sequences(one_hot_vectors, maxlen=30522, dtype="long", truncating="post", padding="post") 
label_ids

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test,s_train,s_test=train_test_split(input_ids,label_ids,sample_weights,test_size=0.2,random_state=0)

# train_classifier_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(config.BATCH_SIZE))

# # We have 25000 examples for testing
# test_classifier_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(config.BATCH_SIZE)

# # Build dataset for end to end model input (will be used at the end)
# test_raw_classifier_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(config.BATCH_SIZE)

def bert_module(query, key, value, i):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=config.NUM_HEAD,
        key_dim=config.EMBED_DIM // config.NUM_HEAD,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


loss_fn = keras.losses.SparseCategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)
loss_tracker = tf.keras.metrics.Mean(name="loss")


class MaskedLanguageModel(tf.keras.Model):
    def train_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            predictions = sample_weight*predictions 
            loss = loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]


def create_masked_language_bert_model():
    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)

    word_embeddings = layers.Embedding(
        config.VOCAB_SIZE, config.EMBED_DIM, name="word_embedding"
    )(inputs)
    position_embeddings = layers.Embedding(
        input_dim=config.MAX_LEN,
        output_dim=config.EMBED_DIM,
        weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
        name="position_embedding",
    )(tf.range(start=0, limit=config.MAX_LEN, delta=1))
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    for i in range(config.NUM_LAYERS):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

    mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(
        encoder_output
    )
    mlm_model = MaskedLanguageModel(inputs, mlm_output, name="masked_bert_model")

    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    mlm_model.compile(optimizer=optimizer)
    return mlm_model


# id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
# token2id = {y: x for x, y in id2token.items()}


class MaskedTextGenerator(keras.callbacks.Callback):
    def __init__(self, sample_tokens, top_k=5):
        self.sample_tokens = sample_tokens
        self.k = top_k

    def decode(self, tokens):
        return " ".join([id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        return id2token[id]

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.sample_tokens)

        masked_index = np.where(self.sample_tokens == mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index
                                        ]

        top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
        values = mask_prediction[0][top_indices]

        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {
                "input_text": self.decode(sample_tokens[0].numpy()),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            pprint(result)


# sample_tokens = vectorize_layer(["I have watched this [mask] and it was awesome"])
# generator_callback = MaskedTextGenerator(sample_tokens.numpy())

# bert_masked_model = create_masked_language_bert_model()
# bert_masked_model.summary()

inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)

word_embeddings = layers.Embedding(
    config.VOCAB_SIZE, config.EMBED_DIM, name="word_embedding"
)(inputs)
position_embeddings = layers.Embedding(
    input_dim=config.MAX_LEN,
    output_dim=config.EMBED_DIM,
    weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
    name="position_embedding",
)(tf.range(start=0, limit=config.MAX_LEN, delta=1))
embeddings = word_embeddings + position_embeddings

encoder_output = embeddings
for i in range(64):
    encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

mlm_output = layers.Dense(128, name="mlm_cls", activation="softmax")(
    encoder_output
)
x=layers.Flatten()(mlm_output)
x=layers.Dense(30522,name='last', activation="softmax")(x)

mlm_model = Model(inputs, x, name="masked_bert_model")

mlm_model.summary()

optimizer = keras.optimizers.Adam(learning_rate=config.LR)
mlm_model.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy",metrics=['acc'])


mlm_model.fit(x_train,y_train,epochs=10, validation_data=(x_test,y_test),verbose=1)

x_train