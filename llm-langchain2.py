import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader
st.set_page_config(page_title= "LLM Model", layout= "wide")
st.title('LLM Model')
padding = 0

st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

def set_bg_hack_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url("https://images.unsplash.com/photo-1519120944692-1a8d8cfc107f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8d2hpdGUlMjB3YWxscGFwZXJ8ZW58MHx8MHx8&auto=format&fit=crop&w=600&q=60");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
set_bg_hack_url()
with st.sidebar: 
    #st.info(main_container )
    #"st.session_state object:",st.session_state
    #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")
    st.title("LLM Model")
    choice = st.radio("Navigation", ["Overview", "LLM Model"])
    st.info("This application demonstrates the use of LLM Model.")

if choice == "Overview":
    pass
if choice == "LLM Model":
    def generate_response(openai_api_key, query_text):
        directory = 'pets'
    
        def load_docs(directory):
          loader = DirectoryLoader(directory)
          documents = loader.load()
          return documents
    
        documents = load_docs(directory)
        #st.info(documents)
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        texts = text_splitter.split_documents(documents)
        #st.info(texts)
        # Select embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return qa.run(query_text)
    
    # Page title
    #st.set_page_config(page_title='🦜🔗 Ask the Doc App')
    st.title('🦜🔗 Ask the Doc App')
    
    
    # Query text
    query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.')
    
    # Form input and query
    result = []
    with st.form('myform', clear_on_submit=True):
        openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (query_text))
        submitted = st.form_submit_button('Submit', disabled=not(query_text))
        if submitted and openai_api_key.startswith('sk-'):
            with st.spinner('Calculating...'):
                response = generate_response(openai_api_key, query_text)
                result.append(response)
                del openai_api_key
    
    if len(result):
        st.info(response)
