from transformers import BertTokenizer, TFBertModel
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

def user_input_language():
    idiom = st.sidebar.selectbox(
        "",
        ([
            "Arabic",
            "Bulgarian",
            "Chinese",
            "English",
            "French",
            "German",
            "Greek",
            "Hindi",
            "Russian",
            "Spanish",
            "Swahili",
            "Thai",
            "Turkish",
            "Urdu",
            "Vietnamese"
            ])
        )
    return idiom

def two_sentences(idiom):
    data = pd.read_csv("project_watson/data/train.csv")
    premise = data.premise[data.language == idiom].reset_index(drop=True)[np.random.randint(0,len(idiom))]
    hypothesis = data[data['premise'] == premise].hypothesis.values[0]
    return premise, hypothesis

premise = premise
hypothesis = hypothesis
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

def encode_sentence(s):
   tokens = list(tokenizer.tokenize(s))
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(hypotheses, premises, tokenizer):

  sentence1 = tf.ragged.constant([encode_sentence(s) for s in np.array(hypotheses)])
  sentence2 = tf.ragged.constant([encode_sentence(s) for s in np.array(premises)])

  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
  input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

  input_mask = tf.ones_like(input_word_ids).to_tensor()

  type_cls = tf.zeros_like(cls)
  type_s1 = tf.zeros_like(sentence1)
  type_s2 = tf.ones_like(sentence2)
  input_type_ids = tf.concat(
      [type_cls, type_s1, type_s2], axis=-1).to_tensor()

  inputs = {
      'input_word_ids': input_word_ids.to_tensor(),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

  return inputs

test_input = bert_encode(premise, hypothesis, tokenizer)
predictions = [np.argmax(i) for i in model.predict(test_input)]

def main():
    #TITLE
    st.write("""
    # Project Watson
    Hello *world!*
    """)
    #SLIDEBAR
    st.sidebar.header("Pick a language")
    idiom = user_input_language()

    #INPUT PARAMETERS
    st.subheader('User Input parameters')
    df_two_sentences = pd.DataFrame(two_sentences(idiom), index=["premise", "hypothesis"]).T
    st.table(df_two_sentences.assign(hack="").set_index("hack"))

    #CLASS LABELS
    st.subheader('Class labels')
    st.write(pd.DataFrame(data={
        "label" : [0,1,2],
        "evaluation" : ["Entailment","Neutral","Contradiction"]
        }).set_index("evaluation"))
    st.write(predictions)

if __name__ == "__main__":
    main()
