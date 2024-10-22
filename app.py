# app.py

from flask import Flask, request, jsonify
from langchain_core.embeddings import FakeEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from uuid import uuid4
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_elasticsearch import SparseVectorStrategy
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import numpy as np
from datasets import load_dataset
from suno import Suno, ModelVersions 


# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

client = Suno(cookie='test', model_version=ModelVersions.CHIRP_V3_5)

print("The client has been initalized and the token is now valid.")

# Configuration from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
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
    api_key=GROQ_API_KEY
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

# # Sample Documents to Add
# documents = [
#     Document(
#         page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
#         metadata={"source": "tweet"},
#     ),
#     Document(
#         page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
#         metadata={"source": "news"},
#     ),
#     Document(
#         page_content="Building an exciting new project with LangChain - come check it out!",
#         metadata={"source": "tweet"},
#     ),
#     Document(
#         page_content="Robbers broke into the city bank and stole $1 million in cash.",
#         metadata={"source": "news"},
#     ),
#     Document(
#         page_content="Wow! That was an amazing movie. I can't wait to see it again.",
#         metadata={"source": "tweet"},
#     ),
#     Document(
#         page_content="Is the new iPhone worth the price? Read this review to find out.",
#         metadata={"source": "website"},
#     ),
#     Document(
#         page_content="The top 10 soccer players in the world right now.",
#         metadata={"source": "website"},
#     ),
#     Document(
#         page_content="LangGraph is the best framework for building stateful, agentic applications!",
#         metadata={"source": "tweet"},
#     ),
#     Document(
#         page_content="The stock market is down 500 points today due to fears of a recession.",
#         metadata={"source": "news"},
#     ),
#     Document(
#         page_content="I have a bad feeling I am going to get deleted :(",
#         metadata={"source": "tweet"},
#     ),
#     Document(
#         page_content="The organization's goal is to make a trillion dollars this year.",
#         metadata={"source": "tweet"},
#     ),
# ]

# # Generate UUIDs for Documents
# uuids = [str(uuid4()) for _ in range(len(documents))]

def initialize_vector_store():
    """
    Initialize the Elasticsearch vector store by adding documents.
    This function checks if the index exists and creates it with mappings if not.
    It then adds documents if the index is empty.
    """
    # TODO this is where wikipedia parsing to RAG dataset will be implemented so we can load our dataset into the Flask Server
    # vector_store.add_documents(documents=documents, ids=uuids)
    
    # Load dataset using the datasets library from Hugging Face
    dataset = load_dataset('rahular/simple-wikipedia', split='train')  # Replace with your actual dataset name
    print("The dataset has been loaded")
    # Iterate through the dataset and convert to LangChain documents
    documents = []
    print("Starting to create the documents")
    i = 0
    for example in dataset:
        i +=1
        page_content = example['text']  # Use 'text' field from the dataset as the content
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
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.2}
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

@app.route('/generate_song', methods=['POST'])
def generate_song():
    """
    API endpoint to generate song lyrics based on user input.
    Expects a JSON payload with a 'query' field.
    """
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    user_query = data['query']
    
    try:
        # Invoke the RAG chain with the user query
        response = chain.invoke(user_query)
        
        # Generate a song songs = 
        songs = client.generate(prompt=response, is_custom=True, wait_audio=True)
        # Download generated songs for song in songs: 

        file_path = ''
        for song in songs:
            file_path = client.download(song=song)
            print(f"Song downloaded to: {file_path}")

        return jsonify({'lyrics': response, 'file_path': str(file_path)})
    
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the server is running.
    """
    return jsonify({'status': 'Server is running.'}), 200

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
