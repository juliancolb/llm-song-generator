from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.llms.ollama import Ollama
from llama_index.core import Document, Settings
from getpass import getpass
from urllib.request import urlopen
import json


# https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#finding-your-cloud-id
ELASTIC_CLOUD_ID = getpass("Elastic Cloud ID: ")

# https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#creating-an-api-key
ELASTIC_API_KEY = getpass("Elastic Api Key: ")

url = "https://raw.githubusercontent.com/elastic/elasticsearch-labs/main/datasets/workplace-documents.json"
response = urlopen(url)
workplace_docs = json.loads(response.read())
metadata = []
content = []
for doc in workplace_docs:
    content.append(doc["content"])
    metadata.append(
        {
            "name": doc["name"],
            "summary": doc["summary"],
            "rolePermissions": doc["rolePermissions"],
        }
    )
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=256
)
docs = text_splitter.create_documents(content, metadatas=metadata)

es_vector_store = ElasticsearchStore(
    es_cloud_id=ELASTIC_CLOUD_ID,
    es_api_key=ELASTIC_API_KEY,
    index_name="workplace_index_elser",
    strategy=SparseVectorStrategy(model_id=".elser_model_2_linux-x86_64"),
)

es_vector_store.add_documents(documents=docs)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = es_vector_store.as_retriever()
template = """Answer the question based only on the following context:\n

                    {context}
                    
                    Question: {question}
                   """
prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke("What are the organizations sales goals?")
