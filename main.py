# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import json
import uuid
from dotenv import load_dotenv
import motor.motor_asyncio
import groq
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_mongodb import MongoDBAtlasVectorSearch
import google.generativeai as genai
import logging
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler that runs on startup and shutdown
    """
    # Startup logic
    try:
        # Create indexes if they don't exist
        await chat_collection.create_index("chat_id", unique=True)
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # Continue even if MongoDB connection fails
        logger.warning("Application continuing without MongoDB connection. Some features may not work.")
    
    yield  # This is where the application runs
    
    # Shutdown logic (if needed)
    logger.info("Application shutting down")

# Initialize MongoDB client
MONGODB_URI = os.getenv("MONGODB_URI")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client.agricultural_chatbot

# Collections
chat_collection = db.chat_histories
vector_collection = db.vector_embeddings
user_preferences = db.user_preferences

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Agricultural RAG Chatbot API",
    description="A FastAPI application for an agricultural chatbot with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = groq.Groq(api_key=GROQ_API_KEY)

# Configure Google Generative AI (for Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Initialize vector store
vector_store = MongoDBAtlasVectorSearch(
    collection=vector_collection,
    embedding=embeddings,
    index_name="vector_index",
)

# Initialize LLM
llm = ChatGroq(
    temperature=0.2,
    model_name="llama3-70b-8192",
    groq_api_key=GROQ_API_KEY,
)

# Define Pydantic models for request/response validation
class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    language: Optional[str] = "english"  # Default language is English

class ChatResponse(BaseModel):
    response: str
    chat_id: str
    timestamp: datetime

class LanguagePreference(BaseModel):
    language: str  # "english" or "tamil"
    user_id: Optional[str] = None

class WebpageRequest(BaseModel):
    url: HttpUrl
    title: Optional[str] = None

class HistoryItem(BaseModel):
    chat_id: str
    created_at: datetime
    language: str
    message_count: int

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

class ChatHistory(BaseModel):
    chat_id: str
    language: str
    messages: List[Message]

# Helper Functions
async def get_language_preference(user_id: str) -> str:
    """
    Get the language preference for a user
    """
    user = await user_preferences.find_one({"user_id": user_id})
    return user["language"] if user else "english"

async def translate_text(text: str, target_language: str) -> str:
    """
    Translate text using Gemini model
    """
    if target_language.lower() not in ["english", "tamil"]:
        return text
    
    prompt = f"Translate the following text to {target_language}:\n\n{text}"
    
    response = genai.GenerativeModel('gemini-pro').generate_content(prompt)
    return response.text

async def create_chat_session():
    """
    Create a new chat session and return the ID
    """
    chat_id = str(uuid.uuid4())
    await chat_collection.insert_one({
        "chat_id": chat_id,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "messages": [],
        "language": "english"
    })
    return chat_id

async def process_pdf(file_path: str, filename: str):
    """
    Process PDF file and store its embeddings
    """
    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                "source": filename,
                "type": "pdf",
                "timestamp": datetime.now().isoformat()
            })
        
        # Split documents
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and store in vector store
        await store_document_embeddings(chunks)
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.info(f"Successfully processed PDF: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {str(e)}")
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        return False

async def process_webpage(url: str, title: Optional[str] = None):
    """
    Process webpage and store its embeddings
    """
    try:
        # Load webpage
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                "source": url,
                "title": title or url,
                "type": "webpage",
                "timestamp": datetime.now().isoformat()
            })
        
        # Split documents
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and store in vector store
        await store_document_embeddings(chunks)
        
        logger.info(f"Successfully processed webpage: {url}")
        return True
    except Exception as e:
        logger.error(f"Error processing webpage {url}: {str(e)}")
        return False

async def store_document_embeddings(documents: List[Document]):
    """
    Store document embeddings in MongoDB Atlas Vector Search
    """
    vector_store.add_documents(documents)
    logger.info(f"Stored {len(documents)} document chunks in vector store")

async def get_rag_response(query: str, language: str):
    """
    Get RAG-enhanced response using LangChain
    """
    # Template for agricultural queries with stronger language instruction
    if language.lower() == "tamil":
        template = """
        You are an agricultural expert assistant helping farmers with their questions.
        Use the following pieces of retrieved context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        IMPORTANT: Your response MUST be COMPLETELY in Tamil language only. No English words should be included.
        Translate all agricultural terms to Tamil equivalents. Provide practical and actionable advice when possible.
        """
    else:
        template = """
        You are an agricultural expert assistant helping farmers with their questions.
        Use the following pieces of retrieved context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Provide practical and actionable advice when possible.
        """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    try:
        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        # Get answer
        result = qa_chain({"query": query})
        response = result["result"]
        
        # For Tamil, add verification step to ensure complete translation
        if language.lower() == "tamil":
            # Double-check that the response is in Tamil by asking the model to translate it fully
            verification_prompt = f"""
            The following is supposed to be entirely in Tamil, but may contain some English words or phrases.
            Please translate any remaining English words or phrases to Tamil, ensuring the entire response is in Tamil only:
            
            {response}
            
            Return the fully Tamil version only.
            """
            
            verification_result = llm.invoke(verification_prompt)
            return verification_result.content
        
        return response
    except Exception as e:
        logger.error(f"Error in RAG response generation: {str(e)}")
        # Fallback to direct LLM call without retrieval
        try:
            if language.lower() == "tamil":
                fallback_prompt = f"""
                நீங்கள் ஒரு விவசாய நிபுணர் உதவியாளராக இருந்து, விவசாயிகளின் கேள்விகளுக்கு உதவுகிறீர்கள்.
                கேள்வி: {query}
                
                செயல்படக்கூடிய மற்றும் செயல்திறனான ஆலோசனைகளை வழங்கவும்.
                """

            else:
                fallback_prompt = f"""
                You are an agricultural expert assistant helping farmers with their questions.
                Question: {query}
                
                Provide practical and actionable advice if possible.
                """
            
            response = llm.invoke(fallback_prompt)
            result = response.content
            
            # For Tamil, verify the response is completely in Tamil
            if language.lower() == "tamil":
                verification_prompt = f"""
                The following is supposed to be entirely in Tamil, but may contain some English words or phrases.
                Please translate any remaining English words or phrases to Tamil, ensuring the entire response is in Tamil only:
                
                {result}
                
                Return the fully Tamil version only be natural dont add statements like -> Here is the fully Tamil version:.
                """
                
                verification_result = llm.invoke(verification_prompt)
                return verification_result.content
            
            return result
        except Exception as fallback_error:
            logger.error(f"Fallback response also failed: {str(fallback_error)}")
            if language.lower() == "tamil":
                return "மன்னிக்கவும், தற்போது பதிலளிப்பதில் சிக்கல் ஏற்பட்டுள்ளது. தயவுசெய்து பின்னர் மீண்டும் முயற்சிக்கவும்."
            else:
                return "I apologize, but I'm having trouble generating a response at the moment. Please try again later."

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Main endpoint for chatbot interaction.
    Accepts a user message and returns a bot response with updated chat history.
    """
    try:
        # Get or create chat session
        chat_id = chat_request.chat_id
        if not chat_id:
            chat_id = await create_chat_session()
            # Set the language for this chat
            await chat_collection.update_one(
                {"chat_id": chat_id},
                {"$set": {"language": chat_request.language}}
            )
        
        # Get chat data
        chat_data = await chat_collection.find_one({"chat_id": chat_id})
        if not chat_data:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        language = chat_request.language or chat_data.get("language", "english")
        
        # Store user message
        user_message = {
            "role": "user",
            "content": chat_request.message,
            "timestamp": datetime.now()
        }
        
        # Get RAG response
        bot_response = await get_rag_response(chat_request.message, language)
        
        # Store bot response
        assistant_message = {
            "role": "assistant",
            "content": bot_response,
            "timestamp": datetime.now()
        }
        
        # Update chat history
        await chat_collection.update_one(
            {"chat_id": chat_id},
            {
                "$push": {
                    "messages": {
                        "$each": [user_message, assistant_message]
                    }
                },
                "$set": {"updated_at": datetime.now(), "language": language}
            }
        )
        
        return ChatResponse(
            response=bot_response,
            chat_id=chat_id,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

@app.post("/language/set")
async def set_language(preference: LanguagePreference):
    """
    Set language preference (English or Tamil)
    """
    if preference.language.lower() not in ["english", "tamil"]:
        raise HTTPException(status_code=400, detail="Supported languages are English and Tamil")
    
    # Generate user ID if not provided
    user_id = preference.user_id or str(uuid.uuid4())
    
    # Update or insert language preference
    await user_preferences.update_one(
        {"user_id": user_id},
        {"$set": {"language": preference.language.lower(), "updated_at": datetime.now()}},
        upsert=True
    )
    
    return {"status": "success", "user_id": user_id, "language": preference.language}

@app.post("/upload/pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload PDF document to knowledge base
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save file temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Process in background
    background_tasks.add_task(process_pdf, file_path, file.filename)
    
    return {"status": "processing", "filename": file.filename}

@app.post("/upload/webpage")
async def upload_webpage(
    background_tasks: BackgroundTasks,
    webpage: WebpageRequest
):
    """
    Upload webpage URL to knowledge base
    """
    # Process in background
    background_tasks.add_task(process_webpage, str(webpage.url), webpage.title)
    
    return {"status": "processing", "url": webpage.url}

@app.get("/history/{chat_id}", response_model=ChatHistory)
async def get_chat_history(chat_id: str):
    """
    Retrieve chat history for a specific conversation
    """
    chat_data = await chat_collection.find_one({"chat_id": chat_id})
    if not chat_data:
        raise HTTPException(status_code=404, detail="Chat history not found")
    
    return ChatHistory(
        chat_id=chat_data["chat_id"],
        language=chat_data.get("language", "english"),
        messages=[
            Message(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"]
            ) for msg in chat_data["messages"]
        ]
    )

@app.get("/history/list", response_model=List[HistoryItem])
async def list_chat_histories():
    """
    List all available chat histories
    """
    histories = []
    async for chat in chat_collection.find().sort("updated_at", -1):
        histories.append(
            HistoryItem(
                chat_id=chat["chat_id"],
                created_at=chat["created_at"],
                language=chat.get("language", "english"),
                message_count=len(chat["messages"])
            )
        )
    return histories

# Add a health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running
    """
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Main function
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)