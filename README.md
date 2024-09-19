# llm-song-generator

Make sure to be in a virtual environment by initializing a venv in the root of your project: 
`python -m venv venv`

On windows, type this to activate your virtual environment: 
`.\venv\Scripts\activate`

Use this command to install the project dependencies: 
`pip install -r requirements.txt`

To run the elastic search vector database example, make sure to have docker installed and use this command to create a local elastic search vector instance that has no authentication: 
`%docker run -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "xpack.security.http.ssl.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.12.1`