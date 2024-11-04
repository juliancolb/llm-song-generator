# app.py

import os
import re
from uuid import uuid4
from flask import Flask, render_template, request, jsonify
#from backend import response

from datasets import load_dataset
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore  # , SparseVectorStrategy
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from suno import ModelVersions, Suno

# import numpy as np
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.llms import Ollama
# from langchain_core.embeddings import FakeEmbeddings
# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

client = Suno(cookie="ajs_anonymous_id=9a6cf472-7fc3-4ae7-9de2-a0462d5dff5d; _ga=GA1.1.821634759.1729025310; _cfuvid=LxAHjXLZ6.IvVp1V4Crxu7a2Lr89ZnTWoQ.Lf8q4LnM-1730387997325-0.0.1.1-604800000; __cf_bm=D8Ol4Q132.5P.YhmL36T_RzjpJzTYt6.zlq6Ygay_gs-1730406940-1.0.1.1-FXV8kc_57Z7dous41X_xSLXzPZxTKdhshFlmpf2zTWN7Pz2Ab8BXqwaKjEjAKm7aUJLkjCUyAEVtxjEXE0Ychg; __client=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImNsaWVudF8yblVRZE9jd0VpWUp3NEREakd5OXc5SjE5N3kiLCJyb3RhdGluZ190b2tlbiI6ImV1eTBsOTA5ZDBoc2htYzh5djNtZ2Jxc2pkM3JpaDhjYmx0eTB2bnIifQ.vpQu-35riqVepIMJBnLaMbfUfvQ661m2lKvTO2bo_zmM_tZgDe8aQ26UN8BIiFisJ-ZnySgwa3H6BMoj-pbivPBlxStgKUpNPhL50eDxnvOtrm1cRMCtZoN8A4b-KkF9pbBNx9LuBi_NKxoY-9X4BlzSWyjurX-9wzchBgYcBGc04UKlFPT-FFQ6LV6-79DHM0nPmfupcuhnJXKhMaH8uAjhanEPuBD514zlX4uifn-gf-7GvmfDX1n7hzDDtWb4E3QV2cBxk4vjcbzvus8rhuIjN9OgnRIafmaxuyM0hgECc13valgmlXIyUEJp7hfRNKRTN6K_Pp7eapbjP7QC0A; __client_uat=1730407463; __client_uat_U9tcbTPE=1730407463; _ga_7B0KEDD7XP=GS1.1.1730407424.4.1.1730407486.0.0.0", model_version=ModelVersions.CHIRP_V3_5)

print("The client has been initalized and the token is now valid.")

# Configuration from environment variables
GROQ_API_KEY = "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"#os.getenv("GROQ_API_KEY")
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = os.getenv("ES_PORT", "9200")
ES_INDEX = os.getenv("ES_INDEX", "langchain-demo")

# Initialize Groq LLM
groq_llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY,
    # Add other parameters if necessary
)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# If you prefer fake embeddings for testing, uncomment the next line
# embeddings = FakeEmbeddings(size=4096)

# Initialize Elasticsearch Vector Store
vector_store = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name=ES_INDEX,
    embedding=embeddings,
    # es_url=f"http://{ES_HOST}:{ES_PORT}"
)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    message = request.form['msg']
    return generate_song(message)


def initialize_vector_store():
    """
    Initialize the Elasticsearch vector store by adding documents.
    This function checks if the index exists and creates it with mappings if not.
    It then adds documents if the index is empty.
    """
    # TODO this is where wikipedia parsing to RAG dataset will be implemented so we can load our dataset into the Flask Server
    # vector_store.add_documents(documents=documents, ids=uuids)

    # Load dataset using the datasets library from Hugging Face
    dataset = load_dataset(
        "rahular/simple-wikipedia", split="train"
    )  # Replace with your actual dataset name
    print("The dataset has been loaded")
    # Iterate through the dataset and convert to LangChain documents
    documents = []
    print("Starting to create the documents")
    i = 0
    for example in dataset:
        i += 1
        page_content = example[
            "text"
        ]  # Use 'text' field from the dataset as the content
        metadata = {}  # If there are other fields to be added as metadata, update this accordingly
        documents.append(Document(page_content=page_content, metadata=metadata))
        if i > 100:
            break

    print("The documents have been created")
    # Generate UUIDs for the new documents
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Add documents to Elasticsearch vector store
    vector_store.add_documents(documents=documents, ids=uuids)
    print("The documents are added to the vector store")


# Initialize the vector store on startup
initialize_vector_store()


# Define the RAG Chain
def create_chain():
    """
    Creates the Retrieval-Augmented Generation (RAG) chain.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2}
    )

    template = """Answer the question based only on the following context:\n

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | groq_llm
        | StrOutputParser()
    )

    return chain


def format_docs(docs):
    """
    Formats the retrieved documents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# Initialize the chain
chain = create_chain()


#@app.route("/generate_song", methods=["POST"])
def generate_song(userMessage):
    """
    API endpoint to generate song lyrics based on user input.
    Expects a JSON payload with a 'query' field.
    """
    #data = request.get_json()

    ##if not data or "query" not in data:
    #    return jsonify({"error": "No query provided"}), 400

    user_query = userMessage

    try:
        # Invoke the RAG chain with the user query
        response = chain.invoke(user_query)
        
        # Generate a single song songs = 
        songs = client.generate(prompt=response, is_custom=True, wait_audio=True)
        # Download generated songs for song in songs: 


        # Generate a song songs =
       # songs = client.generate(prompt=response, is_custom=False, wait_audio=True)
        # Download generated songs for song in songs:
        
        #file_path = os.path.join("static", "generated_song.mp3")  # Path to save in the static directory
        
        for music in songs:
          file_path = client.download(song=music)
          parts = str(music).split()

        # Find the part that starts with "audio_url="
        audio_url = None
        for part in parts:
            if part.startswith("audio_url="):
                # Extract the URL by removing the surrounding quotes
                audio_url = "https://audiopipe.suno.ai/?item_id=" + part.split("=")[2].strip("'")
                break
       
        print(f"Song link:{audio_url}")

        return jsonify({"lyrics": response, "audio_url": str(audio_url) })

    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify(
            {"error": "An error occurred while processing your request."}
        ), 500


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify the server is running.
    """
    return jsonify({"status": "Server is running."}), 200


if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5001, debug=True)
