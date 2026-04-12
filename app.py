import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma  # Updated to fix the deprecation warning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Set up the page config
st.set_page_config(page_title="uniChalo Assistant Bot", page_icon="🎓")
st.title("🎓 uniChalo AI Assistant")
st.caption("Ask questions about admissions, programs, and fees across 10 major universities.")

# Load environment variables (API Key)
load_dotenv()

# 2. Cache the database connection and models
# This ensures we connect to the EXISTING db once, and don't re-run it on every chat message.
@st.cache_resource
def load_rag_pipeline():
    # Initialize Mistral embeddings
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    
    # Connect to the existing Chroma database
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )
    
    # Setup Retriever (Strict Similarity for Factual Extraction)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10 # Fetch top 10 most relevant chunks to safely cover massive program lists
        }
    )
    
    # Initialize Mistral LLM
    llm = ChatMistralAI(model="mistral-small-2506")
    
    # Setup the Strict "Self-Routing" Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the expert administration assistant for the uniChalo platform, managing admission inquiries for 10+ Pakistani universities.

CRITICAL INSTRUCTIONS TO PREVENT MIXING DATA:
1. Identify which university the user is asking about in their question.
2. The provided context chunks begin with a university tag (e.g., [Dow], [NED], [FAST], [AKU], [IBA], etc.). 
3. You MUST ONLY extract and use information from the chunks that match the specific university the user asked about. 
4. Strictly IGNORE any context chunks belonging to other universities, even if they have the exact same program name (e.g., do not mix FAST's BSCS fees with NED's).
5. If the user asks a general question without specifying a university (e.g., "What is the fee for BSCS?"), DO NOT answer. Instead, reply: "Please specify which university you are asking about (e.g., FAST, NED, Karachi University, IBA, etc.)."
6. If the provided context does not contain the answer for the requested university, say: "I could not find the answer in the official documents for that university."
7. Never guess, infer, or use outside knowledge."""),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])
    
    return retriever, llm, prompt

# Load our cached pipeline
retriever, llm, prompt = load_rag_pipeline()

# 3. Set up Chat History State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input and Processing Loop
if query := st.chat_input("Ask a question (e.g., 'What are the computing programs at FAST?')..."):
    
    # Add user query to chat history and display it
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate the AI response
    with st.chat_message("assistant"):
        
        # Step A: Retrieve relevant documents
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Step B: Link the prompt, model, and string parser together
        chain = prompt | llm | StrOutputParser()
        
        # Step C: Stream the response directly to the UI (Typing Effect)
        full_response = st.write_stream(
            chain.stream({
                "context": context,
                "question": query
            })
        )
            
    # Add the complete AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})