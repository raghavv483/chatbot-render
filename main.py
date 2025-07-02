import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import other libraries with error handling
try:
    from sentence_transformers import SentenceTransformer
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain
    import chromadb
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.stop()

# ------------------- CONFIG -------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CSV_PATH = "./walmart_products.csv"

# Get API key with better error handling
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # Try Streamlit secrets as fallback
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except:
        st.error("❌ Google API Key not found! Please set it in Streamlit secrets or environment variables.")
        st.stop()

# ------------------- STREAMLIT CONFIG -------------------
st.set_page_config(
    page_title="Walmart Product Assistant",
    page_icon="🛒",
    layout="wide"
)

# ------------------- CACHE FUNCTIONS -------------------
@st.cache_resource
def load_model():
    """Load the sentence transformer model with error handling"""
    try:
        with st.spinner("Loading embedding model..."):
            return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_data
def load_data():
    """Load CSV data with better error handling"""
    try:
        # Check if file exists
        if not os.path.exists(CSV_PATH):
            st.error(f"❌ CSV file not found at {CSV_PATH}")
            return None, None, None
        
        with st.spinner("Loading product data..."):
            df = pd.read_csv(CSV_PATH)
            
            # Validate required columns
            required_cols = ['name', 'brand', 'category', 'price', 'discount', 
                           'description', 'stock_quantity', 'store_location']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns in CSV: {missing_cols}")
                return None, None, None
            
            # Create documents
            documents = []
            metadatas = []
            
            for _, row in df.iterrows():
                doc = (f"Product: {row.name}, Brand: {row.brand}, Category: {row.category}, "
                      f"Price: ₹{row.price}, Discount: {row.discount}%, Description: {row.description}, "
                      f"Stock: {row.stock_quantity} units, Store: {row.store_location}")
                documents.append(doc)
                
                meta = {"brand": str(row.brand), "category": str(row.category), "store": str(row.store_location)}
                metadatas.append(meta)
            
            return df, documents, metadatas
            
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None, None, None

@st.cache_resource
def setup_chromadb():
    """Setup ChromaDB with error handling"""
    try:
        data_result = load_data()
        if data_result[0] is None:
            return None
            
        df, documents, metadatas = data_result
        model = load_model()
        
        if model is None:
            return None
        
        with st.spinner("Setting up vector database..."):
            # Create embeddings
            embeddings = model.encode(documents).tolist()
            
            # Initialize ChromaDB client
            client = chromadb.Client()
            collection = client.get_or_create_collection(name="walmart_products")
            
            # Clear existing data and add new data
            if collection.count() > 0:
                collection.delete(ids=[f"prod_{i}" for i in range(collection.count())])
            
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=[f"prod_{i}" for i in range(len(documents))]
            )
            
            return collection
            
    except Exception as e:
        st.error(f"Failed to setup ChromaDB: {e}")
        return None

@st.cache_resource
def setup_llm():
    """Setup LLM with error handling"""
    try:
        with st.spinner("Initializing AI model..."):
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3
            )
    except Exception as e:
        st.error(f"Failed to setup LLM: {e}")
        return None

@st.cache_resource
def setup_rag_chain():
    """Setup RAG chain with error handling"""
    try:
        llm = setup_llm()
        if llm is None:
            return None
            
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
        
    except Exception as e:
        st.error(f"Failed to setup RAG chain: {e}")
        return None

# ------------------- MAIN APP -------------------
def main():
    st.title("🛒 Walmart Product Assistant")
    st.markdown("Ask me anything about Walmart products!")
    
    # Show loading progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components with progress tracking
        status_text.text("🔄 Loading embedding model...")
        progress_bar.progress(20)
        model = load_model()
        if model is None:
            st.stop()
        
        status_text.text("🔄 Loading product data...")
        progress_bar.progress(40)
        data_result = load_data()
        if data_result[0] is None:
            st.stop()
        df, _, _ = data_result
        
        status_text.text("🔄 Setting up vector database...")
        progress_bar.progress(60)
        collection = setup_chromadb()
        if collection is None:
            st.stop()
        
        status_text.text("🔄 Initializing AI assistant...")
        progress_bar.progress(80)
        rag_chain = setup_rag_chain()
        if rag_chain is None:
            st.stop()
        
        status_text.text("✅ Ready!")
        progress_bar.progress(100)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
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
                    
                    if not results["documents"][0]:
                        st.warning("No matching products found for your query.")
                        return
                    
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
        
        # Debug information
        with st.expander("🔍 Debug Information"):
            st.write("**Environment Variables:**")
            st.write(f"- GOOGLE_API_KEY exists: {bool(GOOGLE_API_KEY)}")
            st.write(f"- CSV file exists: {os.path.exists(CSV_PATH)}")
            st.write(f"- Current directory: {os.getcwd()}")
            st.write(f"- Files in directory: {os.listdir('.')}")

if __name__ == "__main__":
    main()