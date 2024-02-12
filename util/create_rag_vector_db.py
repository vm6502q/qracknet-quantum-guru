# See https://plainenglish.io/community/super-quick-retrieval-augmented-generation-using-ollama-6078c1

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders import PyPDFLoader

from langchain.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings import OllamaEmbeddings

import os

DB_PATH = "vectorstores/db/"

def create_vector_db():
    documents=[]
    processed_pdfs=0
    processed_txts=0
    for f in os.listdir("data"):
        try:
            if f.endswith(".pdf"):
                pdf_path = './data/' + f
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
                processed_pdfs+=1
            elif f.endswith(".txt"):
                _f = open('./data/', 'r')
                documents.extend(_f.read())
                _f.close()
                processed_txts+=1
        except:
            print("issue with ", f)
            pass
    print("Processed ", processed_pdfs, " pdf files, ", processed_txts, " txt files")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts=text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(),persist_directory=DB_PATH)
    vectorstore.persist()

if __name__=="__main__":
    create_vector_db()
