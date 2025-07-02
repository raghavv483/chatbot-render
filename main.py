import streamlit as st
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

# ------------------- STREAMLIT CONFIG -------------------
st.set_page_config(
    page_title="Walmart Product Assistant",
    page_icon="🛒",
    layout="wide"
)

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
def setup_chromadb():
    df, documents, metadatas = load_data()
    model = load_model()
    
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
    
    return collection

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
        collection = setup_chromadb()
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
                    
                    # Search in ChromaDB
                    results = collection.query(
                        query_embeddings=[query_embedding],
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