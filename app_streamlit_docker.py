import requests
import json
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# CSS = """
# body {
#     background-image: url(https://cdn.pixabay.com/photo/2017/08/30/01/05/milky-way-2695569_960_720.jpg);
#     background-size: cover;
# }
# """
# st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

# def user_input_language():
#     idiom = st.sidebar.selectbox(
#         "",
#         ([
#             "Arabic",
#             "Bulgarian",
#             "Chinese",
#             "English",
#             "French",
#             "German",
#             "Greek",
#             "Hindi,
#             "Russian",
#             "Spanish",
#             "Swahili",
#             "Thai",
#             "Turkish",
#             "Urdu",
#             "Vietnamese"
#             ])
#         )
    # return idiom

# def two_sentences(idiom):
#     data = pd.read_csv("project_watson/data/train.csv")
#     premise = data.premise[data.language == idiom].reset_index(drop=True)[np.random.randint(0,len(idiom))]
#     hypothesis = data[data['premise'] == premise].hypothesis.values[0]
#     return premise, hypothesis

def encode_sentence(s):
    """ Encode One sentence s using the tokenizer defined in this .py"""
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(hypotheses, premises, tokenizer):
    """Returns a mathematical representation of the text inputs using the defined
    tokenizer. The model expects 'input_word_ids', 'input_mask', 'input_type_ids'"""

    sentence1 = tf.ragged.constant([encode_sentence(s) for s in np.array(hypotheses)])
    sentence2 = tf.ragged.constant([encode_sentence(s) for s in np.array(premises)])

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
    # take the 1st 50 tokens of the 20 first lines and convert to list. Why 20 ? No reason, could be 1...
    token_num = test_input['input_word_ids'].shape[1]
    if token_num >= 50 :
        test_input_slice = {'input_word_ids': tf.slice(test_input['input_word_ids'], [0, 0], [20, 50]).numpy().tolist(),
                            'input_mask' : tf.slice(test_input['input_mask'], [0, 0], [20, 50]).numpy().tolist(),
                            'input_type_ids' : tf.slice(test_input['input_type_ids'], [0, 0], [20, 50]).numpy().tolist()}
    else:
        test_input_slice = {'input_word_ids': pad_sequences(test_input["input_word_ids"], maxlen=50, padding='post').tolist(),
                            'input_mask': pad_sequences(test_input["input_mask"], maxlen=50, padding='post').tolist(),
                            'input_type_ids': pad_sequences(test_input["input_type_ids"], maxlen=50, padding='post').tolist()}
    #format data as expected by the model for a prediction AND KEEP ONLY THE 1ST LINE (TO DO : refactor)
    # ==> {'instances': [{'input_word_ids': [101, 10111, ... 11762] , 'input_mask': [1, 1, ... 1], 'input_type_ids': [0, 0, ... 0]

    data = {}
    for k in test_input_slice.keys():
        data[k] = test_input_slice[k][0]

    data = {'instances': [data]}

    return data

def predict(data, url='http://34.78.186.53/v1/models/bert-base:predict'):
    response = requests.post(
        url=url,
        headers={
            "Host": "bert-base-model.default.example.com"
        },
        data=json.dumps(data)
    )
    return response.json()['predictions'][0]

def main():

    #TITLE
    # st.markdown(f'<style>{CSS}</style>', unsafe_allow_html=True)
    # st.(<style> h1{color: blue;}"Contradictory, my dear Watson"</style>, unsafe_allow_html=True)
    #SLIDEBAR
    # st.sidebar.header("Start with picking a language")
    # idiom = user_input_language()
    # CLASS LABELS
    st.sidebar.subheader('Class labels')
    st.sidebar.write(pd.DataFrame(data={
        "label" : [0,1,2],
        "evaluation" : ["Entailment","Neutral","Contradiction"]
        }).set_index("evaluation"))

    #INPUT PARAMETERS
    st.subheader('Write your statements below!')
    user_premise = st.text_area("Your premise here")
    user_hypothesis = st.text_area("Your hypothesis here")
    data = {'premise': [user_premise], 'hypothesis': [user_hypothesis]}
    df_two_sentences = pd.DataFrame(data=data)
    st.table(df_two_sentences)

    test = df_two_sentences
    embedded_s = format_input(test, tokenizer)

    predictions = predict(embedded_s)
    st.write(predictions)

    # st.write(pd.DataFrame(data=embedded_s))

if __name__ == "__main__":
    model_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    main()
