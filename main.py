from langchain .document_loaders import PyPDFLoader,TextLoader,UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings
from langchain.llms import OpenAI
import tiktoken
import chromadb
import os

persist_directory = 'db'
embeddings = OpenAIEmbeddings()
encoding = tiktoken.encoding_for_model('davinci')
tokenizer = tiktoken.get_encoding(encoding.name)

def tk_len(text):
    token = tokenizer.encode (
        text,
        disallowed_special=()
    )
    return len(token)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tk_len,
    separators=['\n\n','\n',',','']
)

def Embeddings(chunks):
    vectordb = Chroma.from_documents(chunks,embeddings,persist_directory=persist_directory)
    vectordb.persist()


def Files(filenames):
    for file in filenames:
        _,file_extension = os.path.splitext(file)
        if file_extension == ".pdf":
            loader = PyPDFLoader(file)
            docs = loader.load()
        elif file_extension == ".txt":
            loader = TextLoader(file)
            docs = loader.load()
        elif file_extension == ".docx" or file_extension == ".doc":
            loader = UnstructuredWordDocumentLoader(file)
            docs = loader.load()
        chunks = text_splitter.split_documents(docs)
        Embeddings(chunks)

def Delete_files(filename):
    metadata = {}
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="db"
    ))
    collection_name = client.list_collections()[0].name
    collection = client.get_collection(name=collection_name)
    metadata['source'] = filename
    collection.delete(
        where=metadata
    )    


def response(query):
    vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    assist = RetrievalQA.from_llm(OpenAI(temperature=0, model_name="text-davinci-003"),
                                                retriever=vectordb.as_retriever(kwargs={'2'}))
    response = assist(query)
    return response['result']
