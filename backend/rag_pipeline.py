import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# We need the user to have provided GROQ_API_KEY
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize embeddings (HuggingFace Open Source)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VECTOR_STORE_PATH = "faiss_index"

def get_vector_store():
    # If it exists, load it, otherwise create a new empty one (or handle gracefully)
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def ingest_document(file_path: str, filename: str):
    if filename.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
        
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Add to or create vector store
    vectorstore = get_vector_store()
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        vectorstore.add_documents(chunks)
        
    vectorstore.save_local(VECTOR_STORE_PATH)

def ask_question(query: str) -> str:
    if not groq_api_key:
        return "Error: GROQ_API_KEY not found in environment variables."

    vectorstore = get_vector_store()
    if not vectorstore:
        return "No documents have been indexed yet. Please upload a document."
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatGroq(temperature=0.2, model_name="llama-3.1-8b-instant")
    
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(query)
