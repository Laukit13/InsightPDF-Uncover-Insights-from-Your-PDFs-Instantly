import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("PDF Question and Answer Prj By LM")
st.sidebar.title("HEllO")

pdf = st.file_uploader("Upload your PDF", type='pdf', accept_multiple_files=True)
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

process_url_clicked = st.sidebar.button("pdf")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    loader = PyPDFLoader(pdf)
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    time.sleep(2)

    # Save the FAISS index to a pickle file
    vectorstore_openai.save_local("faiss_store")

query = main_placeholder.text_input("Question: ")
if query:
    vectorIndex = FAISS.load_local("faiss_store", OpenAIEmbeddings())
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
    result = chain.invoke({"question": query}, return_only_outputs=True)
    # result will be a dictionary of this format --> {"answer": "", "sources": [] }
    st.header("Answer")
    st.write(result["answer"])

    # Display sources, if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)
