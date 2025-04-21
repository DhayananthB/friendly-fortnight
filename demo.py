import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model_name="llama3-8b-8192", api_key=GROQ_API_KEY)

prompt = "What is the capital of France?"
response = llm.invoke(prompt)  # ✅ just a plain string

print(response)
