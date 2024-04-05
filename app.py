import os
from dotenv import load_dotenv
from src.helper import repo_ingestion
from src.helper import load_embedding
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import  ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from flask import Flask, render_template, jsonify, request


app= Flask(__name__)


load_dotenv()
OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')

embeddings= load_embedding()

persist_directory= 'db'

# Now we are loading persisted database from disk, use it as normal.
vectordb= Chroma(persist_directory= persist_directory, 
                    embedding_function= embeddings)

llm= ChatOpenAI()

memory= ConversationSummaryMemory(llm= llm, memory_key='chat_history', return_messages=True)

qa_chain= ConversationalRetrievalChain.from_llm(llm= llm,
                                       retriever= vectordb.as_retriever(search_type='mmr', search_kwargs={"k":3}),
                                       memory= memory)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/chatbot', methods=['GET', 'POST'])
def gitRepo():
    if request.method== 'POST':
        user_input= request.form["question"]
        repo_ingestion(user_input)
        os.system("python store_index.py")
    return jsonify({"response": str(f"cloned this repository {user_input}")})


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg= request.form["msg"]
    input= msg
    print(input)
    
    if input.lower()== "clear":
        os.system("rm -rf repo")
        return "Previously Cloned Repository is deleted"
    else:
        result= qa_chain.invoke(input)
        print(result['answer'])
        return str(result['answer'])
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
    
    
