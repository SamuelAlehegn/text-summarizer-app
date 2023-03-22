from nlp import nlp_app
import altair as alt
from rouge import Rouge
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


def evaluate(summary, reference):
    r = Rouge()
    scores = r.get_scores(summary, reference)
    scores_df = pd.DataFrame(scores[0])
    return scores_df


def summy(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


def main():
    st.title("Text Summarizer App")
    menu = ["Home", "NLP", "About"]
    choice = st.sidebar.selectbox("Menu", menu, key="1")

    if choice == "Home":
        st.subheader("TextRank Summary vs LexRank Summary")
        raw_text = st.text_area("Your Text", "Type Here")
        if st.button("Summarize"):
            with st.expander("Original Text"):
                st.write(raw_text)

            c1, c2 = st.columns(2)
            with c1:
                with st.expander("LexRank Summary"):
                    summary = summy(raw_text)
                    document_length = {'Original Length': len(
                        raw_text), 'Summary Length': len(summary)}
                    st.write(document_length)
                    st.write(summary)

                    st.info("Rouge Score")
                    score = evaluate(summary, raw_text)
                    st.dataframe(score)

                    # Plotting the Rouge Score
                    score['matrics'] = score.index
                    c = alt.Chart(score).mark_bar().encode(
                        x='matrics', y='rouge-1')
                    st.altair_chart(c)

            with c2:
                with st.expander("TextRank Summary"):
                    summary = summarize(raw_text)
                    document_length = {'Original Length': len(
                        raw_text), 'Summary Length': len(summary)}
                    st.write(document_length)
                    st.write(summary)

                    st.info("Rouge Score")
                    score = evaluate(summary, raw_text)
                    st.dataframe(score)

                    # Plotting the Rouge Score
                    score['matrics'] = score.index
                    c = alt.Chart(score).mark_bar().encode(
                        x='matrics', y='rouge-1')
                    st.altair_chart(c)

    elif choice == "NLP":
        # st.subheader("Natural Language Processing")
        nlp_app()

    elif choice == "About":
        st.subheader("About")
        st.write("A text summarizer and natural language processing project built using Python Streamlit is a tool designed to help users quickly and easily summarize large pieces of text. The tool uses natural language processing techniques to analyze and extract the most important information from the text, allowing users to quickly get a sense of the key points without having to read through the entire document. ")
        st.write("The user interface is built using Streamlit, a popular open-source framework for building data science and machine learning applications. The interface is designed to be intuitive and easy to use, with a simple text input field where users can paste in the text they want to summarize.")
        st.write("Whenever a user submits their content, the program analyzes it using a variety of methods, including clustering, sentence scoring, and keyword extraction, to extract the most crucial information. The final summary, as well as any pertinent information like word count and reading duration, are shown in a separate output field.")
        st.write("The tool is designed to be easy to use and understand, and it can be used by anyone who wants to quickly summarize a large piece of text. It can be used by students, journalists, and anyone else who needs to quickly get a sense of the key points in a document without having to read through the entire thing.")


if __name__ == "__main__":
    main()
