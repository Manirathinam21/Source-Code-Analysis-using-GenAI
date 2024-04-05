import os
from git import Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language
from langchain.embeddings import OpenAIEmbeddings

def repo_ingestion(repo_url):
    os.makedirs("repo", exist_ok=True)
    repo_path= "repo/"
    Repo.clone_from(repo_url, to_path=repo_path)
    

def load_repo(repo_path):
    loader =GenericLoader.from_filesystems(repo_path, 
                          glob= "**/*",
                          suffixes=[".py"],
                          parser= LanguageParser(language=Language.PYTHON, parser_threshold=500)
                          )
    documents= loader.load()
    return documents

# Creating Text Chunks
def text_splitter(documents):
    text_splitter= RecursiveCharacterTextSplitter.from_language(language= Language.PYTHON,
                                                 text_chunks= 2000,
                                                 text_overlap=200)  
    
    text_chunks= text_splitter.split_documents(documents)
    return text_chunks

# Loading Embeddings Model
def load_embedding():
    embedding= OpenAIEmbeddings(disallowed_special=())
    return embedding
