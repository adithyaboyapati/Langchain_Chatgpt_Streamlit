# Langchain_Chatgpt_Streamlit

Streamlit app that was built to answer the query from the given set of documents with the help of Langchhain and OPENAI API.

Steps involved:
1. Loading the documents from the folder
2. Reading the context from the documents and splitting the text into chunks
3. These chunks are passed into the Embedding model to convert the text to vectors
4. The embeddings along with the original documents are stored in the vector databases for faster retrieval of information
5. Initialize OPEN API key
6. When we ask the query the query will be passed to the vector database to see the most relevent information to the query
7. The relevant information that was retrieved was passed to the OPENAI API to detect the context from the information and answer the query
based on the context 
