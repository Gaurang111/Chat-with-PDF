# !pip install langchain
# !pip install openai
# !pip install PyPDF2 # To load and reaf Pdf files
# !pip install faiss-cpu
"""
Faiss is a library for efficient similarity search and clustering of dense vectors.
It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.
It also contains supporting code for evaluation and parameter tuning.
Faiss is written in C++ with complete wrappers for Python/numpy.
It is developed by Facebook AI Research.
"""
# !pip install tiktoken

from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

import os
import streamlit as st


OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

def process_pdf_file(pdffile):
    try:
        # provide the path of pdf file/files.
        pdfreader = PdfReader(pdffile)

        # read text from pdf
        raw_text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content

        # print(raw_text)

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=50,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()

        document_search = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(llm, chain_type="stuff")

        query=st.text_input("Write your query here...")
        # query = "what is the age group for vaccination of girls for prevention of cervical cancer"
        # query = "In burning Process of turritella and Bentonite, what is Industrial Furnace?"
        # query = "In Mix Proportion of Self Compacting Concrete (SCC), what is Target Mean Strength formula?"
        if query:
            docs = document_search.similarity_search(query)
            st.write(chain.run(input_documents=docs, question=query))

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.title("Chat with your PDF...")
pdffile = st.file_uploader("Upload your PDFs here:", type="pdf")

if pdffile is not None:
    process_pdf_file(pdffile)
else:
    st.warning("Please upload a PDF file.")
