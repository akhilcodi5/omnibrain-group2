import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0
)

response = llm.invoke("Explain LangGraph in one sentence.")

print(response.content)