import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Sidebar contents
with st.sidebar:
    st.title('Chat With Your PDFs 💬')
    st.markdown('''
    Tools Used:
    - Streamlit - UI design
    - LangChain - LLM framework
    - [OpenAI]- LLM model
    ''')

load_dotenv()
def main():
    st.header("Find Answers From Your pdfs")

    pdf_docs = st.file_uploader("Upload your PDF", type='pdf', accept_multiple_files=True)

    if pdf_docs is not None:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        docs = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_texts(docs, embedding=embeddings)
        # vectorstore_openai.save_local("faiss_store")
        # vectorIndex = FAISS.load_local("faiss_store", OpenAIEmbeddings())

        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)

        if query:
            docs = vectorstore_openai.similarity_search(query=query, k=2)

            llm = OpenAI(temperature=0.9, max_tokens=500)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == '__main__':
    main()
