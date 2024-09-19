import json
from urllib.request import urlopen

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_elasticsearch import ElasticsearchStore, SparseVectorStrategy


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
    es_url="http://localhost:9201",
    index_name="workplace_index_elser",
    strategy=SparseVectorStrategy(model_id=".elser_model_2_linux-x86_64"),
)

es_vector_store.add_documents(documents=docs)

llm = Ollama(model="llama3")


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
