from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma  # Updated to fix the deprecation warning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (MISTRAL_API_KEY)
load_dotenv()

app = FastAPI(title="uniChalo RAG API")

# Enable CORS so your Node backend or Frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Connecting to existing database...")

# 1. Initialize the embedding model
embedding_model = MistralAIEmbeddings(model="mistral-embed")

# 2. Connect to the EXISTING Chroma database folder
vectorstore = Chroma(
    persist_directory="chroma_db", 
    embedding_function=embedding_model
)

# 3. Set up the retriever (Updated to match your Streamlit logic: k=10)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 10 # Fetch top 10 most relevant chunks to safely cover massive program lists
    }
)

# 4. Initialize the LLM
llm = ChatMistralAI(model="mistral-small-2506")

# 5. Set up the Strict "Self-Routing" uniChalo Prompt
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

# 6. Chain the prompt, model, and output parser together
chain = prompt | llm | StrOutputParser()

print("uniChalo API is ready to receive questions!")

# Define the expected format of incoming requests
class QueryRequest(BaseModel):
    query: str

# Create the API endpoint
@app.post("/api/chatbot")
async def chat_endpoint(request: QueryRequest):
    try:
        # Step A: Retrieve relevant chunks from the database
        docs = retriever.invoke(request.query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Step B: Generate the answer using the chain
        response = chain.invoke({
            "context": context, 
            "question": request.query
        })
        
        # Step C: Return the answer as pure JSON to your Node.js backend
        return {"answer": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))