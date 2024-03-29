# See https://plainenglish.io/community/super-quick-retrieval-augmented-generation-using-ollama-6078c1

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.ollama import OllamaEmbeddings

import os

DB_PATH = "vectorstores/db/"

def create_vector_db():
    documents=[]
    texts=[]
    processed_pdfs=0
    processed_txts=0
    for f in os.listdir("data"):
        if f.endswith(".pdf"):
            pdf_path = "./data/" + f
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            processed_pdfs+=1
        elif f.endswith(".txt") or f.endswith(".md") or f.endswith(".hpp") or f.endswith(".cpp"):
            _f = open("./data/" + f, 'r')
            texts.extend(_f.read())
            _f.close()
            processed_txts+=1
    print("Processed ", processed_pdfs, " pdf files, ", processed_txts, " txt files")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    if processed_txts > 0:
        texts = text_splitter.create_documents(texts)
        documents += texts
    texts=text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=texts, embedding=OllamaEmbeddings(),persist_directory=DB_PATH)
    vectorstore.persist()

if __name__=="__main__":
    create_vector_db()
