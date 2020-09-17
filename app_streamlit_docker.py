import requests
import json
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#CSS STYLE
CSS = """
body {
    background-image: url(https://images.pexels.com/photos/3109168/pexels-photo-3109168.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260);
    background-size: cover;
}

.block-container {
    background-color: rgba(0,0,0,.4);
}

.block-container div {
    # background-color: background-color: rgba(0,0,0,.6);
}

.element-container {
    # background-color: blue;
}

h1 {
    text-align: center;
    font-family: "Trebuchet MS", Helvetica, sans-serif;
    color: #EEE8AA;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}

h2 {
    font-family: "Trebuchet MS", Helvetica, sans-serif;
    color: #FFE4E1;
    font-weight: bold;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}

h3 {
    font-family: "Trebuchet MS", Helvetica, sans-serif;
    font-weight: bold;
    color: white;
}

.fullScreenFrame :nth-child(1) {
    background-color: white;
}

"""

st.write(f'<style>{CSS}</style>',
    unsafe_allow_html=True)

def encode_sentence(s, tokenizer):
    """ Encode One sentence s using the tokenizer defined in this .py"""
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(hypotheses, premises, tokenizer):
    """Returns a mathematical representation of the text inputs using the defined
    tokenizer. The model expects 'input_word_ids', 'input_mask', 'input_type_ids'"""

    sentence1 = tf.ragged.constant([encode_sentence(s, tokenizer) for s in np.array(hypotheses)])
    sentence2 = tf.ragged.constant([encode_sentence(s, tokenizer) for s in np.array(premises)])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {'input_word_ids': input_word_ids.to_tensor(),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

    return inputs

def format_input(test, tokenizer):
    """Reduces #tokens to 50 (max_len), as expected by the model to predict."""
    test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)
    token_num = test_input['input_word_ids'].shape[1]

    test_input_slice = {'input_word_ids': pad_sequences(test_input["input_word_ids"], maxlen=50, padding='post', truncating='post').tolist(),
                        'input_mask': pad_sequences(test_input["input_mask"], maxlen=50, padding='post', truncating='post').tolist(),
                        'input_type_ids': pad_sequences(test_input["input_type_ids"], maxlen=50, padding='post', truncating='post').tolist()}

    data = {}
    for k in test_input_slice.keys():
        data[k] = test_input_slice[k][0]

    data = {'instances': [data]}

    return data

def choose_model():
    st.markdown("""
        ### Select a model
        """)

    model = st.selectbox("",["BERT", "roBERTa"],1)

    if model == 'BERT':
        model_name = 'bert-base-multilingual-cased'
    else:
        model_name = 'roberta-large'

    return model_name

#BERT
def predict_bert(data, url="http://104.155.26.32/v1/models/bert-base:predict"):
    response = requests.post(
        url=url,
        headers={
            "Host": "bert-base.default.example.com"
        },
        data=json.dumps(data)
    )
    return response.json()['predictions'][0]

#ROBERTA
def predict_roberta(data, url="http://104.155.26.32/v1/models/roberta:predict"):
    response = requests.post(
        url=url,
        headers={
            "Host": "roberta.default.example.com"
        },
        data=json.dumps(data)
    )
    return response.json()['predictions'][0]

def main():
    #TITLE
    st.markdown("""
        # Contradictory, My Dear Watson!
        """)

    #USER INPUT
    st.markdown("""
        ### Your premise:
        """)
    user_premise = st.text_area("")
    st.markdown("""
        ### Your hypothesis:
        """)
    user_hypothesis = st.text_area(" ")
    data = {'premise': [user_premise], 'hypothesis': [user_hypothesis]}
    df_two_sentences = pd.DataFrame(data=data)

    model_name = choose_model()

    #PREDICTIONS
    embedded_s = format_input(df_two_sentences, AutoTokenizer.from_pretrained(model_name))

    if user_hypothesis != "":
        #BERT
        if model_name == "bert-base-multilingual-cased":
            df_proba = pd.DataFrame(predict_bert(embedded_s), columns=["probability"]).rename({0: "Entailment", 1: "Neutral", 2: "Contradiction"}, axis='index').round(2)
        #ROBERTA
        else:
            embedded_s["instances"][0]["input_type_ids"] = 50*[0]
            df_proba = pd.DataFrame(predict_roberta(embedded_s), columns=["probability"]).rename({0: "Entailment", 1: "Neutral", 2: "Contradiction"}, axis='index').round(2)

        st.write(f""" ## The predicted relationship is **{df_proba.idxmax()[0]}** :male-detective:""")

        if st.checkbox("show probabilities"):
            st.write(df_proba.style.background_gradient(cmap='magma').highlight_max(color="#AEE3AA", axis=0))


if __name__ == "__main__":
    main()
