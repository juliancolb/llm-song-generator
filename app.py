# app.py

import os
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

client = Suno(cookie="__client=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImNsaWVudF8yblVRZE9jd0VpWUp3NEREakd5OXc5SjE5N3kiLCJyb3RhdGluZ190b2tlbiI6Imp4djR1MnQ5MzU2Z2o2czlmcnllaHdxbTI0NWE4M3g5ZGR6MnpwYTcifQ.j1nMDW9YuwM2wPaTrDAcWn6C23aAYs89vTssFhbO3bRyklRY0T3-LB_0F_xjCG-RmBTPcAwfqlGZDK3a7Hm3sHnxCdh0gBoeYQY2k_zR0BwQv9SQ0xI1ey2dB0wAWQ7BuWbz46BZtBF-IWoKAsjk3WP2tZRynRG_G2Cdp9tYBR3SBaLeZ2ulA7Wstj0T9PQO2K6ht4DTcmPZ3ud_ZD6cwILYh9zj78o0J0aQ7lqMMHDsjoV5eKdWaLb7fdiGQU8wT6dLO-aGqQ0ATfSZO8seAkF3NCrDa7Exjl9VW-Lh2qc13NwRvGyYt6cX6FXziYy63KhW1IWgPU6PZ16DxsUzUQ; __client_uat=1729025288; __client_uat_U9tcbTPE=1729025288; ajs_anonymous_id=9a6cf472-7fc3-4ae7-9de2-a0462d5dff5d; _ga=GA1.1.821634759.1729025310; __cf_bm=bKkN05_IAcEh6svoyzNjna4l2GLMy1FKObcAff8dudc-1730235404-1.0.1.1-WHuPJXsMdlJ67HVM_CX4gDqld4mL7zd.2dd_70QDHh7YVt0nF61TV.mzmjBZmsoMLfJsQ7zMVjwo5Ev0Sdok9g; _cfuvid=JiQ4UcbErQxEoZlAOjVyjzAor5CZYuoYBKECGA6Hb_8-1730235404296-0.0.1.1-604800000; _ga_7B0KEDD7XP=GS1.1.1730235405.2.1.1730235463.0.0.0", model_version=ModelVersions.CHIRP_V3_5)

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
        
        # Generate a song songs = 
        songs = client.generate(prompt=response, is_custom=True, wait_audio=True)
        # Download generated songs for song in songs: 

        # Generate a song songs =
        songs = client.generate(prompt=response, is_custom=False, wait_audio=True)
        # Download generated songs for song in songs:

        file_path = ""
        for song in songs:
            file_path = client.download(song=song)
            print(f"Song downloaded to: {file_path}")

        return jsonify({"lyrics": response, "file_path": str(file_path)})

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
