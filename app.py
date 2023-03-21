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
    choice = st.sidebar.selectbox("Menu", menu)

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
        st.subheader("Natural Language Processing")

    elif choice == "About":
        st.subheader("About")


if __name__ == "__main__":
    main()
