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
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from pymongo import DeleteMany

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
    # Create scheduler
    scheduler = AsyncIOScheduler()
    
    # Startup logic
    try:
        # Create indexes if they don't exist
        await chat_collection.create_index("chat_id", unique=True)
        
        # Initialize vector store during startup
        global vector_store
        vector_store = await init_vector_store()
        
        # Create index for websites collection
        await websites_collection.create_index("url", unique=True)
        
        # Schedule daily scraping at 1 AM
        scheduler.add_job(
            run_scheduled_scraping,
            CronTrigger(hour=1, minute=0),  # Run at 1:00 AM every day
            id="daily_scraping"
        )
        
        # Start the scheduler
        scheduler.start()
        logger.info("Scheduled tasks initialized")
        
        # Add the fertilizer website if it doesn't exist
        default_website = {
            "url": "http://115.243.209.84/ARS/fert_stock_position/index/en",
            "title": "Fertilizer Stock Position",
            "description": "Current fertilizer stock data from government portal",
            "scrape_type": "table",
            "active": True,
            "created_at": datetime.now()
        }
        
        await websites_collection.update_one(
            {"url": default_website["url"]},
            {"$set": default_website},
            upsert=True
        )
        
        # Initial scraping of all websites
        background_tasks = BackgroundTasks()
        background_tasks.add_task(run_scheduled_scraping)
        
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # Continue even if MongoDB connection fails
        logger.warning("Application continuing without MongoDB connection. Some features may not work.")
    
    yield  # This is where the application runs
    
    # Shutdown logic
    if scheduler.running:
        scheduler.shutdown()
    logger.info("Application shutting down")

# Initialize MongoDB client
MONGODB_URI = os.getenv("MONGODB_URI")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client.agricultural_chatbot

# Collections
chat_collection = db.chat_histories
vector_collection = db.vector_embeddings
user_preferences = db.user_preferences
websites_collection = db.websites_to_scrape

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
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Initialize vector store - this will be properly initialized in the lifespan startup
vector_store = None

# Function to initialize vector store
async def init_vector_store():
    """
    Initialize and return the MongoDB Atlas Vector Search store
    """
    # Ensure the collection exists
    collection_names = await db.list_collection_names()
    if vector_collection.name not in collection_names:
        await db.create_collection(vector_collection.name)
    
    # Initialize vector store with a properly wrapped collection to handle async operations
    import asyncio
    from pymongo.collection import Collection
    
    # Patch the collection to avoid the RuntimeWarning
    # This removes the async list_collection_names and create_collection calls from inside MongoDBAtlasVectorSearch
    class SyncCollection(Collection):
        def __init__(self, async_collection):
            self._async_collection = async_collection
            super().__init__(
                async_collection.database.delegate, 
                async_collection.name,
                create=False
            )
    
    sync_collection = SyncCollection(vector_collection)
    
    store = MongoDBAtlasVectorSearch(
        collection=sync_collection,
        embedding=embeddings,
        index_name="vector_index",
        relevance_score_fn="cosine",
    )
    
    # Create vector search index - handle creation in async-aware way
    try:
        # MongoDB Atlas Vector Search doesn't have a specific async method for creating index
        # Create the index synchronously but handle it properly in async context
        await asyncio.to_thread(
            lambda: store.create_vector_search_index(dimensions=768)
        )
        logger.info("Vector search index created or updated successfully")
    except Exception as e:
        logger.warning(f"Vector search index might already exist: {str(e)}")
    
    return store

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

class ScrapableWebsite(BaseModel):
    url: HttpUrl
    title: str
    description: Optional[str] = None
    scrape_type: str = "auto"  # "auto", "table", "json", "text", "custom"
    selector: Optional[str] = None  # CSS selector for finding the element to scrape
    json_path: Optional[str] = None  # JSONPath expression for JSON data extraction
    headers: Optional[Dict[str, str]] = None  # Custom headers for the request
    active: bool = True
    last_scraped: Optional[datetime] = None

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
    # Use asyncio to run the synchronous add_documents method in a thread pool
    import asyncio
    await asyncio.to_thread(lambda: vector_store.add_documents(documents))
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

async def scrape_website(website: Dict[str, Any]):
    """
    Scrape data from a website based on its scrape_type
    """
    try:
        url = website["url"]
        title = website["title"]
        scrape_type = website.get("scrape_type", "auto")
        
        # If scrape_type is auto, try to detect the content type
        if scrape_type == "auto":
            scrape_type = await detect_content_type(url, website.get("headers"))
        
        # Call appropriate scraper based on type
        if scrape_type == "json":
            return await scrape_json_from_website(website)
        elif scrape_type == "table":
            return await scrape_table_from_website(website)
        elif scrape_type == "text":
            return await scrape_text_from_website(website)
        else:
            logger.error(f"Unsupported scrape type: {scrape_type} for {url}")
            return False
    except Exception as e:
        logger.error(f"Error determining scrape type for {website['url']}: {str(e)}")
        return False

async def detect_content_type(url: str, headers: Optional[Dict[str, str]] = None):
    """
    Detect the content type of a URL (JSON or HTML with tables)
    """
    try:
        # Use custom headers if provided
        headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # Send a HEAD request first to check content type
        head_response = requests.head(url, headers=headers)
        content_type = head_response.headers.get('Content-Type', '').lower()
        
        if 'application/json' in content_type:
            return "json"
        
        # If not json by header, try to fetch and analyze
        response = requests.get(url, headers=headers)
        
        # Check if it begins with JSON structure
        text = response.text.strip()
        if text.startswith('{') and text.endswith('}') or text.startswith('[') and text.endswith(']'):
            try:
                json.loads(text)
                return "json"
            except json.JSONDecodeError:
                pass
        
        # Check for tables in HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table')
        if tables:
            return "table"
        
        # Default to text if no other format detected
        return "text"
    except Exception as e:
        logger.error(f"Error detecting content type: {str(e)}")
        # Default to table as fallback
        return "table"

async def scrape_json_from_website(website: Dict[str, Any]):
    """
    Scrape JSON data from a website API endpoint
    """
    try:
        import jsonpath_ng
        from jsonpath_ng import parse
        
        url = website["url"]
        title = website["title"]
        json_path = website.get("json_path")
        headers = website.get("headers", {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        # Send GET request
        response = requests.get(url, headers=headers)
        
        # Try to parse JSON
        if response.status_code != 200:
            logger.error(f"Failed to fetch JSON from {url}: Status code {response.status_code}")
            return False
        
        json_data = response.json()
        
        # Extract specific data if json_path is provided
        if json_path:
            jsonpath_expr = parse(json_path)
            matches = [match.value for match in jsonpath_expr.find(json_data)]
            # Use extracted data if found, otherwise use full JSON
            if matches:
                data_to_store = matches
            else:
                data_to_store = json_data
        else:
            data_to_store = json_data
        
        # Convert to string representation
        if isinstance(data_to_store, list) and len(data_to_store) > 0:
            # If it's a list of objects, try to convert to DataFrame then CSV
            try:
                df = pd.DataFrame(data_to_store)
                data_string = df.to_csv(index=False)
                data_type = "tabular_json"
            except Exception:
                data_string = json.dumps(data_to_store, indent=2)
                data_type = "json"
        else:
            # Otherwise just pretty-print the JSON
            data_string = json.dumps(data_to_store, indent=2)
            data_type = "json"
        
        # Create document for vector store
        document = Document(
            page_content=f"{title} Data:\n{data_string}",
            metadata={
                "source": url,
                "title": title,
                "type": "scraped_data",
                "format": data_type,
                "timestamp": datetime.now().isoformat(),
                "description": website.get("description", "Scraped JSON data")
            }
        )
        
        # Delete previous versions of this data
        await vector_collection.delete_many({
            "metadata.source": url,
            "metadata.type": "scraped_data"
        })
        
        # Split if necessary
        chunks = text_splitter.split_documents([document])
        
        # Store embeddings
        await store_document_embeddings(chunks)
        
        # Update last scraped timestamp
        await websites_collection.update_one(
            {"url": url},
            {"$set": {"last_scraped": datetime.now()}}
        )
        
        logger.info(f"Successfully scraped and stored JSON data from: {url}")
        return True
    except Exception as e:
        logger.error(f"Error scraping JSON data from {website['url']}: {str(e)}")
        return False

async def scrape_table_from_website(website: Dict[str, Any]):
    """
    Scrape table data from a website and store it in the knowledge base
    """
    try:
        url = website["url"]
        title = website["title"]
        selector = website.get("selector")
        headers = website.get("headers", {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        # Send GET request
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'  # Ensure proper character decoding
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table - use selector if provided, otherwise find first table
        if selector:
            table = soup.select_one(selector)
        else:
            table = soup.find('table')
        
        if not table:
            logger.error(f"No table found on the page: {url}")
            return False
            
        # Use pandas to read the HTML table
        df_list = pd.read_html(str(table))
        if not df_list:
            logger.error(f"Could not parse table data from: {url}")
            return False
            
        df = df_list[0]
        
        # Convert to string and add metadata
        csv_string = df.to_csv(index=False)
        
        # Create document for vector store
        document = Document(
            page_content=f"{title} Data:\n{csv_string}",
            metadata={
                "source": url,
                "title": title,
                "type": "scraped_data",
                "format": "table",
                "timestamp": datetime.now().isoformat(),
                "description": website.get("description", "Scraped tabular data")
            }
        )
        
        # Delete previous versions of this data
        await vector_collection.delete_many({
            "metadata.source": url,
            "metadata.type": "scraped_data"
        })
        
        # Split if necessary
        chunks = text_splitter.split_documents([document])
        
        # Store embeddings
        await store_document_embeddings(chunks)
        
        # Update last scraped timestamp
        await websites_collection.update_one(
            {"url": url},
            {"$set": {"last_scraped": datetime.now()}}
        )
        
        logger.info(f"Successfully scraped and stored table data from: {url}")
        return True
    except Exception as e:
        logger.error(f"Error scraping table data from {url}: {str(e)}")
        return False

async def scrape_text_from_website(website: Dict[str, Any]):
    """
    Scrape text content from a website
    """
    try:
        url = website["url"]
        title = website["title"]
        selector = website.get("selector")
        headers = website.get("headers", {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        # Send GET request
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text content
        if selector:
            elements = soup.select(selector)
            if not elements:
                logger.error(f"No elements found with selector '{selector}' on {url}")
                return False
            
            text_content = "\n".join(element.get_text(strip=True) for element in elements)
        else:
            # Get main content or fallback to body
            main = soup.find('main') or soup.find('article') or soup.find('body')
            text_content = main.get_text(separator="\n", strip=True)
        
        if not text_content.strip():
            logger.error(f"No text content extracted from {url}")
            return False
        
        # Create document for vector store
        document = Document(
            page_content=f"{title}:\n{text_content}",
            metadata={
                "source": url,
                "title": title,
                "type": "scraped_data",
                "format": "text",
                "timestamp": datetime.now().isoformat(),
                "description": website.get("description", "Scraped text content")
            }
        )
        
        # Delete previous versions
        await vector_collection.delete_many({
            "metadata.source": url,
            "metadata.type": "scraped_data"
        })
        
        # Split document into chunks
        chunks = text_splitter.split_documents([document])
        
        # Store embeddings
        await store_document_embeddings(chunks)
        
        # Update last scraped timestamp
        await websites_collection.update_one(
            {"url": url},
            {"$set": {"last_scraped": datetime.now()}}
        )
        
        logger.info(f"Successfully scraped and stored text from: {url}")
        return True
    except Exception as e:
        logger.error(f"Error scraping text from {url}: {str(e)}")
        return False

async def run_scheduled_scraping():
    """
    Run scraping for all active websites in the database
    """
    websites = await websites_collection.find({"active": True}).to_list(length=100)
    logger.info(f"Running scheduled scraping for {len(websites)} websites")
    
    for website in websites:
        await scrape_website(website)
        
    logger.info("Scheduled scraping completed")

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

@app.post("/websites/add", response_model=ScrapableWebsite)
async def add_website_to_scrape(website: ScrapableWebsite):
    """
    Add a website to the list of websites to scrape regularly
    """
    # Convert the model to a dictionary
    website_dict = website.model_dump()
    
    # Add creation timestamp
    website_dict["created_at"] = datetime.now()
    
    # Insert into collection
    await websites_collection.update_one(
        {"url": str(website.url)},
        {"$set": website_dict},
        upsert=True
    )
    
    # Return the created website
    return website

@app.get("/websites/list", response_model=List[ScrapableWebsite])
async def list_websites_to_scrape():
    """
    List all websites registered for scraping
    """
    websites = await websites_collection.find().to_list(length=100)
    return websites

@app.post("/websites/scrape-now/{website_id}")
async def scrape_website_now(website_id: str, background_tasks: BackgroundTasks):
    """
    Manually trigger scraping for a specific website
    """
    from bson.objectid import ObjectId
    
    website = await websites_collection.find_one({"_id": ObjectId(website_id)})
    if not website:
        raise HTTPException(status_code=404, detail="Website not found")
    
    # Process in background
    background_tasks.add_task(scrape_website, website)
    
    return {"status": "processing", "website": website["url"]}

@app.post("/websites/scrape-all-now")
async def scrape_all_websites_now(background_tasks: BackgroundTasks):
    """
    Manually trigger scraping for all active websites
    """
    background_tasks.add_task(run_scheduled_scraping)
    return {"status": "processing"}

# Main function
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)