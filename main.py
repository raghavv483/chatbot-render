import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# ------------------- CONFIG -------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CSV_PATH = "./walmart_products.csv"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ------------------- STREAMLIT CONFIG -------------------
st.set_page_config(
    page_title="Walmart Product Assistant",
    page_icon="🛒",
    layout="wide"
)

# ------------------- VECTOR STORE CLASS -------------------
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
    
    def add(self, documents, embeddings, metadatas, ids):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
    
    def query(self, query_embedding, n_results=5, where=None):
        # Filter by metadata if specified
        if where:
            filtered_indices = []
            for i, metadata in enumerate(self.metadatas):
                match = True
                for key, value in where.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_indices.append(i)
        else:
            filtered_indices = list(range(len(self.documents)))
        
        if not filtered_indices:
            return {"documents": [[]], "distances": [[]]}
        
        # Calculate similarities
        filtered_embeddings = [self.embeddings[i] for i in filtered_indices]
        similarities = cosine_similarity([query_embedding], filtered_embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = {
            "documents": [[self.documents[filtered_indices[i]] for i in top_indices]],
            "distances": [[1 - similarities[i] for i in top_indices]]
        }
        
        return results

# ------------------- CACHE FUNCTIONS -------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_data
def load_data():
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
    
    return df, documents, metadatas

@st.cache_resource
def setup_vector_store():
    df, documents, metadatas = load_data()
    model = load_model()
    
    embeddings = model.encode(documents).tolist()
    
    vector_store = SimpleVectorStore()
    vector_store.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[f"prod_{i}" for i in range(len(documents))]
    )
    
    return vector_store

@st.cache_resource
def setup_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

@st.cache_resource
def setup_rag_chain():
    llm = setup_llm()
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
    return LLMChain(llm=llm, prompt=prompt)

# ------------------- MAIN APP -------------------
def main():
    st.title("🛒 Walmart Product Assistant")
    st.markdown("Ask me anything about Walmart products!")
    
    # Initialize components
    try:
        model = load_model()
        vector_store = setup_vector_store()
        rag_chain = setup_rag_chain()
        df, _, _ = load_data()
        
        # Get unique store locations
        stores = df['store_location'].unique().tolist()
        
        # Sidebar for store selection
        with st.sidebar:
            st.header("🏪 Store Selection")
            selected_store = st.selectbox("Choose a store:", stores, index=0)
            
            st.header("📊 Store Stats")
            store_df = df[df['store_location'] == selected_store]
            st.metric("Total Products", len(store_df))
            st.metric("Categories", store_df['category'].nunique())
            st.metric("Brands", store_df['brand'].nunique())
        
        # Main chat interface
        st.header(f"💬 Chat with {selected_store} Store Assistant")
        
        # Chat input
        user_question = st.text_input("Ask your question:", placeholder="What products do you have in electronics?")
        
        if st.button("Ask") and user_question:
            with st.spinner("Searching for products..."):
                try:
                    # Get query embedding
                    query_embedding = model.encode([user_question])[0]
                    
                    # Search in vector store
                    results = vector_store.query(
                        query_embedding=query_embedding,
                        n_results=5,
                        where={"store": selected_store}
                    )
                    
                    context_chunks = results["documents"][0]
                    context = "\n".join(context_chunks)
                    
                    # Generate answer
                    answer = rag_chain.run({
                        "context": context,
                        "question": user_question
                    })
                    
                    # Display results
                    st.success("Found matching products!")
                    st.write("**Answer:**")
                    st.write(answer)
                    
                    # Show context in expander
                    with st.expander("📋 Product Details Found"):
                        for i, chunk in enumerate(context_chunks, 1):
                            st.write(f"**Product {i}:**")
                            st.write(chunk)
                            st.divider()
                    
                    st.info(f"Found {len(context_chunks)} matching products")
                    
                except Exception as e:
                    st.error(f"Error processing your question: {str(e)}")
        
        # Display sample products
        st.header(f"🛍️ Sample Products from {selected_store}")
        store_df = df[df['store_location'] == selected_store].head(5)
        
        for _, product in store_df.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.subheader(product['name'])
                    st.write(f"**Brand:** {product['brand']}")
                    st.write(f"**Category:** {product['category']}")
                    st.write(f"**Description:** {product['description'][:100]}...")
                
                with col2:
                    st.metric("Price", f"₹{product['price']}")
                    st.write(f"**Discount:** {product['discount']}%")
                
                with col3:
                    st.metric("Stock", f"{product['stock_quantity']} units")
                
                st.divider()
    
    except Exception as e:
        st.error(f"Error initializing the app: {str(e)}")
        st.write("Please check your configuration and try again.")

if __name__ == "__main__":
    main()