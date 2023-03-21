import neattext.functions as nfx
import neattext as nt
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
import altair as alt
from rouge import Rouge
from gensim.summarization import summarize
import spacy
nlp = spacy.load('en_core_web_sm')

matplotlib.use('Agg')


def analayzer(raw_text):
    docx = nlp(raw_text)
    tokens = [token.text for token in docx]
    allData = [[
        token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        token.shape_, token.is_alpha, token.is_stop
    ] for token in docx]
    df = pd.DataFrame(allData, columns=[
        'token text', 'lemma', 'part of speech', 'tag', 'dependency',
        'shape', 'alpha', 'stopword'
    ])

    return df


def nlp_app():
    menu = ['Home', 'NLP', 'About']

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == 'Home':
        st.subheader("Analyse Your Text")
        raw_text = st.text_area("Your Text", "Type Here")
        # num_sentences = st.sidebar.number_input("Number of Sentences", 1, 20)
        if st.button("Analyse"):
            # st.write(raw_text)
            with st.expander("Original Text"):
                st.write(raw_text)

            with st.expander("Analysis"):
                token_result_df = analayzer(raw_text)
                st.dataframe(token_result_df)

            with st.expander("Entities"):
                st.write(raw_text)

            column1, column2 = st.columns(2)

            with column1:
                with st.expander("statistics"):
                    st.write(raw_text)
                with st.expander("Top Keywords"):
                    st.write(raw_text)
                with st.expander("Sentiment"):
                    st.write(raw_text)

            with column2:
                with st.expander("Word Frrquency"):
                    st.write(raw_text)
                with st.expander("Part of Speech"):
                    st.write(raw_text)
                with st.expander("Word Cloud"):
                    st.write(raw_text)
            with st.expander("Download Results"):
                st.write(raw_text)

    elif choice == 'NLP':
        st.subheader("Natural Language Processing")
    else:
        st.subheader("About")


if __name__ == '__main__':
    nlp_app()
