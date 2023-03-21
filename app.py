from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')


def summy(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


def main():
    st.title("Text Summarizer App")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        raw_text = st.text_area("Your Text", "Type Here")
        if st.button("Summarize"):
            with st.expander("Original Text"):
                st.write(raw_text)

            c1, c2 = st.columns(2)
            with c1:
                with st.expander("LexRank Summary"):
                    summary = summy(raw_text)
                    st.write(summary)
            with c2:
                with st.expander("TextRank Summary"):
                    summary = summarize(raw_text)
                    st.write(summary)

    elif choice == "About":
        st.subheader("About")


if __name__ == "__main__":
    main()
