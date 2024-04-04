import os
from dotenv import load_env
from src.helper import repo_ingestion, load_repo, text_splitter, load_embedding
from langchain.vectorstores import Chroma
from langchain.memory import ConvertionSummaryMemory
from langchain.chains import ConversationalRetrievalChain


# Loading OpenAI API_KEY
load_env()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


documents= load_repo("repo/")
text_chunks= text_splitter(documents)
embeddings= load_embedding()

# convert codes into vectors and storing in ChromaDB
vectordb= Chroma.from_documents(text_chunks,
                                embedding= embeddings,
                                persist_directory='./db')
vectordb.persist()

