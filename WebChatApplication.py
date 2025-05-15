# Import necessary libraries
import streamlit as st
from decouple import config
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_together.embeddings import TogetherEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain



# Initialize environment variables
GROQ_API_KEY = config("GROQ_API_KEY")
TOGATHER_API_KEY = config("TOGATHER_API_KEY")

# App configuration
st.set_page_config(page_title="Website Chat Assistant", page_icon="🌐")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Page layout
st.title("🌐 Website Chat Assistant")
st.caption("Chat with any website like a Personal Assistance")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    website_url = st.text_input("Enter Website URL", placeholder="https://example.com")
    
    if st.button("Process Website"):
        if not website_url:
            st.error("Please enter a valid URL")
        else:
            with st.spinner("Processing website..."):
                try:
                    # Configure headers and loader
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
                    }

                    loader = WebBaseLoader(
                        web_paths=(website_url,),
                        header_template=headers,
                        requests_kwargs={"verify": False}
                    )
                    
                    # "Step - 01: Load and process website content"
                    docs = loader.load()
        
                    if not docs or not docs[0].page_content.strip():
                        raise ValueError("Received empty content from website")
                    
                    
                    # Document processing
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
        
                    # "Step - 02: Split documents"
                    splits = text_splitter.split_documents(docs)
                    
                    # Initialize embeddings with Together AI API key and model
                    embeddings = TogetherEmbeddings(
                        api_key=TOGATHER_API_KEY,  # Replace with your Together AI API key
                        model="BAAI/bge-base-en-v1.5"
                    )

                    # Create FAISS vector store from your document splits
                    vector_store = FAISS.from_documents(splits, embeddings)
                    # "Step - 03: Create vector store"

                    # Store vector store in session state (if using Streamlit or similar)
                    st.session_state.vector_store = vector_store
                    st.session_state.processed = True
                    # "Step - 04: Store vector store in session state"                    
                    
                    # Show preview
                    st.subheader("Website Preview")
                    preview_text = docs[0].page_content[:500] + "..." if docs else ""
                    st.text(preview_text)
                    st.success("Website processed successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing website: {str(e)}")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.processed:
    if prompt := st.chat_input("Ask about the website..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    
                    # Initialize Groq LLM with DeepSeek model
                    llm = ChatGroq(
                        api_key=GROQ_API_KEY,  # Replace with your Groq API key
                        model_name="deepseek-r1-distill-llama-70b",  # DeepSeek model on Groq
                        temperature=0.3,
                        streaming=True,
                        max_tokens=512
                    )

                    

                    # Check if vector store and message state exists
                    if "vector_store" not in st.session_state or "messages" not in st.session_state:
                        st.error("Session not properly initialized. Please refresh and try again.")
                        st.stop()

                    # Setup retriever
                    retriever = st.session_state.vector_store.as_retriever()

                    # Optimized system prompt — concise, intent-focused, markdown-safe
                    prompt_template = ChatPromptTemplate.from_template(
                        """You are a highly efficient, context-aware assistant. 
                    Respond succinctly with clear and relevant information using only the provided context. 
                    If context is missing, politely state that you don't have enough information.

                    <context>
                    {context}
                    </context>

                    User: {input}
                    Assistant:"""
                    )

                    # Build document and retrieval chains
                    document_chain = create_stuff_documents_chain(llm, prompt_template)
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    # "Step - 05: Build document and retrieval chains"

                    try:
                        # Perform retrieval and generation
                        response = retrieval_chain.invoke({"input": prompt})
                        answer = response.get("answer", "⚠️ No answer returned.")

                        # split the answer from <think> ... </think> and get the content after the tags
                        answer = answer.split("</think>")[1]

                        # Display and store the assistant's response with streaming 
                        st.markdown(answer)
                        # st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                    except Exception as e:
                        st.error(f"🚨 An error occurred while processing your request: {str(e)}")

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
else:
    st.info("Please enter a website URL and click 'Process Website' to start chatting")
