# from wordcloud import WordCloud
from PyPDF2 import PdfFileReader, PdfReader
import pdfplumber
import docx2txt
import time
import base64
from textblob import TextBlob
import collections as counter
import streamlit.components.v1 as stc
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


def download(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = 'nlp_result_{}_.csv'.format(time.strftime('%Y%m%d-%H%M%S'))
    st.markdown("#### Download File ###")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
    st.markdown(href, unsafe_allow_html=True)


def analayzer(raw_text):
    docx = nlp(raw_text)
    tokens = [token.text for token in docx]
    allData = [[
        token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        token.shape_, token.is_alpha, token.is_stop
    ] for token in docx]
    df = pd.DataFrame(allData, columns=[
        'token text', 'lemma', 'part of speech', 'tag', 'dependency',
        'shape', 'alpha', 'stop word'
    ])
    return df


def entities(raw_text):
    docx = nlp(raw_text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem font-color:white;">{}</div>"""


def render_entities(raw_text):
    docx = nlp(raw_text)
    html = spacy.displacy.render(docx, style="ent")
    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)
    return result


def keyword(raw_text):

    word_freq = counter.Counter(raw_text.split())
    common_words = dict(word_freq.most_common(5))
    # df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
    return common_words


def sentiment(raw_text):
    blob = TextBlob(raw_text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        result = "Positive"
    elif sentiment_score < 0:
        result = "Negative"
    else:
        result = "Neutral"
    return result


# def read_pdf(pdf_file):
#     pdfReader = PdfReader(pdf_file)
#     count = pdfReader.numPages
#     text = ""
#     for i in range(count):
#         page = pdfReader.getPage(i)
#         text += page.extractText()
#     return text

def read_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        return text


def nlp_app():
    menu = ['Text Summarizer', 'NLP']

    choice = st.sidebar.selectbox("Menu", menu, key='2')
    if choice == 'Text Summarizer':
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
                # result = entities(raw_text)
                # st.write(result)
                result = render_entities(raw_text)
                stc.html(result, height=100, scrolling=True)

            column1, column2 = st.columns(2)

            with column1:
                with st.expander("statistics"):
                    st.info("Word Statistics")
                    docx = nt.TextFrame(raw_text)
                    st.write(docx.word_stats())

                with st.expander("Top Keywords"):
                    st.info("Top Keywords")
                    processed_text = nfx.remove_stopwords(raw_text)
                    top_keyword = keyword(processed_text)
                    st.write(top_keyword)

                with st.expander("Sentiment"):
                    st.write(sentiment(raw_text))

            with column2:
                with st.expander("Plot Word Frrquency"):
                    fig = plt.figure(figsize=(10, 10))
                    sns.countplot(x='token text',
                                  data=token_result_df)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                with st.expander("Part of Speech"):
                    fig = plt.figure(figsize=(10, 10))
                    sns.countplot(x='part of speech',
                                  data=token_result_df)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                with st.expander("Word Cloud"):
                    st.write(raw_text)
            with st.expander("Download Results"):
                download(token_result_df)

    elif choice == 'NLP':
        st.subheader("Natural Language Processing")
        text_file = st.file_uploader(
            "Upload Text File", type=['txt', 'pdf', 'docx'])
        if text_file is not None:
            if text_file.type == 'application/pdf':
                raw_text = read_pdf(text_file)

                with st.expander("Original Text"):
                    st.write(raw_text)

                with st.expander("Analysis"):
                    token_result_df = analayzer(raw_text)
                    st.dataframe(token_result_df)

                with st.expander("Entities"):
                    # result = entities(raw_text)
                    # st.write(result)
                    result = render_entities(raw_text)
                    stc.html(result, height=100, scrolling=True)

                column1, column2 = st.columns(2)

                with column1:
                    with st.expander("statistics"):
                        st.info("Word Statistics")
                        docx = nt.TextFrame(raw_text)
                        st.write(docx.word_stats())

                    with st.expander("Top Keywords"):
                        st.info("Top Keywords")
                        processed_text = nfx.remove_stopwords(raw_text)
                        top_keyword = keyword(processed_text)
                        st.write(top_keyword)

                    with st.expander("Sentiment"):
                        st.write(sentiment(raw_text))

                with column2:
                    with st.expander("Plot Word Frrquency"):
                        fig = plt.figure(figsize=(10, 10))
                        sns.countplot(x='token text',
                                      data=token_result_df)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    with st.expander("Part of Speech"):
                        fig = plt.figure(figsize=(10, 10))
                        sns.countplot(x='part of speech',
                                      data=token_result_df)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

                    with st.expander("Word Cloud"):
                        st.write(raw_text)
                with st.expander("Download Results"):
                    download(token_result_df)
            elif text_file.type == 'text/plain':
                raw_text = str(text_file.read(), 'utf-8')

                with st.expander("Original Text"):
                    st.write(raw_text)

                with st.expander("Analysis"):
                    token_result_df = analayzer(raw_text)
                    st.dataframe(token_result_df)

                with st.expander("Entities"):
                    # result = entities(raw_text)
                    # st.write(result)
                    result = render_entities(raw_text)
                    stc.html(result, height=100, scrolling=True)

                column1, column2 = st.columns(2)

                with column1:
                    with st.expander("statistics"):
                        st.info("Word Statistics")
                        docx = nt.TextFrame(raw_text)
                        st.write(docx.word_stats())

                    with st.expander("Top Keywords"):
                        st.info("Top Keywords")
                        processed_text = nfx.remove_stopwords(raw_text)
                        top_keyword = keyword(processed_text)
                        st.write(top_keyword)

                    with st.expander("Sentiment"):
                        st.write(sentiment(raw_text))

                with column2:
                    with st.expander("Plot Word Frrquency"):
                        fig = plt.figure(figsize=(10, 10))
                        sns.countplot(x='token text',
                                      data=token_result_df)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    with st.expander("Part of Speech"):
                        fig = plt.figure(figsize=(10, 10))
                        sns.countplot(x='part of speech',
                                      data=token_result_df)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

                    with st.expander("Word Cloud"):
                        st.write(raw_text)
                with st.expander("Download Results"):
                    download(token_result_df)

            else:
                raw_text = docx2txt.process(text_file)

                with st.expander("Original Text"):
                    st.write(raw_text)

                with st.expander("Analysis"):
                    token_result_df = analayzer(raw_text)
                    st.dataframe(token_result_df)

                with st.expander("Entities"):
                    # result = entities(raw_text)
                    # st.write(result)
                    result = render_entities(raw_text)
                    stc.html(result, height=100, scrolling=True)

                column1, column2 = st.columns(2)

                with column1:
                    with st.expander("statistics"):
                        st.info("Word Statistics")
                        docx = nt.TextFrame(raw_text)
                        st.write(docx.word_stats())

                    with st.expander("Top Keywords"):
                        st.info("Top Keywords")
                        processed_text = nfx.remove_stopwords(raw_text)
                        top_keyword = keyword(processed_text)
                        st.write(top_keyword)

                    with st.expander("Sentiment"):
                        st.write(sentiment(raw_text))

                with column2:
                    with st.expander("Plot Word Frrquency"):
                        fig = plt.figure(figsize=(10, 10))
                        sns.countplot(x='token text',
                                      data=token_result_df)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    with st.expander("Part of Speech"):
                        fig = plt.figure(figsize=(10, 10))
                        sns.countplot(x='part of speech',
                                      data=token_result_df)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

                    with st.expander("Word Cloud"):
                        st.write(raw_text)
                with st.expander("Download Results"):
                    download(token_result_df)

    elif choice == 'About':
        st.subheader("About")


if __name__ == '__main__':
    nlp_app()
