from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import chromadb

# ------------------- CONFIG -------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CSV_PATH = "./walmart_products.csv"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PORT = int(os.getenv("PORT", 8000))  # Get PORT from .env with fallback to 8000

# ------------------- FASTAPI SETUP -------------------
app = FastAPI()

# Optional: Allow cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- REQUEST MODEL -------------------
class Query(BaseModel):
    question: str
    store: str = "Jaipur"

# ------------------- LOAD CSV DATA -------------------
df = pd.read_csv(CSV_PATH)

documents = [
    f"Product: {row.name}, Brand: {row.brand}, Category: {row.category}, "
    f"Price: ₹{row.price}, Discount: {row.discount}%, Description: {row.description}, "
    f"Stock: {row.stock_quantity} units, Store: {row.store_location}"
    for _, row in df.iterrows()
]

metadatas = [
    {"brand": row.brand, "category": row.category, "store": row.store_location}
    for _, row in df.iterrows()
]

# ------------------- EMBEDDINGS + CHROMADB -------------------
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(documents).tolist()

client = chromadb.Client()
collection = client.get_or_create_collection(name="walmart_products")

if collection.count() == 0:
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[f"prod_{i}" for i in range(len(documents))]
    )

# ------------------- Gemini LLM -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# ------------------- Prompt + RAG Chain -------------------
template = """
You are a helpful assistant for Walmart store data.
Use the following context to answer the user's question.
Answer in 1-2 lines by greeting the customer with meow.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)
rag_chain = LLMChain(llm=llm, prompt=prompt)

# ------------------- /ask ENDPOINT -------------------
@app.post("/ask")
def ask_question(query: Query):
    query_embedding = model.encode([query.question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"store": query.store}
    )

    context_chunks = results["documents"][0]
    context = "\n".join(context_chunks)

    answer = rag_chain.run({
        "context": context,
        "question": query.question
    })

    return {
        "answer": answer,
        "store": query.store,
        "matches_found": len(context_chunks)
    }

# Optional health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Root endpoint
@app.get("/")
def root():
    return {"message": "Walmart RAG API is running"}

# ------------------- RENDER DEPLOYMENT SUPPORT -------------------
# For local development and production deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)