# chatbot_app.py
import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ============ CONFIG ============
genai.configure(api_key="AIzaSyCmuOyyZRF6BJZCK0g0Z-EHl07WAqQCuQs")  
MODEL = "gemini-2.0-flash-lite"
EMBED_MODEL = "models/embedding-001"
EMBEDDINGS_FILE = "embeddings.json"
# ================================

# ---- Load embeddings ----
@st.cache_data
def load_embeddings(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = [item["chunk"] for item in data]
    embeddings = np.array([item["embedding"] for item in data])
    return chunks, embeddings

chunks, embeddings = load_embeddings(EMBEDDINGS_FILE)

# ---- Search for relevant chunks ----
def retrieve_relevant_chunks(query, chunks, embeddings, top_k=10):
    query_embed = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query"
    )["embedding"]
    similarities = cosine_similarity([query_embed], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# ---- Ask Gemini with context ----
def ask_gemini(query, context_chunks):
    context_text = "\n\n".join(context_chunks)
    prompt = f"""
Give me answer in clear and details also give the process in nepali language. 
Give only in nepali Language, and provide link only if is available in context.
Answer only based on the context, if context not available answer 'answer not available in nepali language'

Context:  
{context_text}

User Question:  
{query}

Answer:
"""
    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(prompt)
    return response.text if response else "⚠️ No response from Gemini."

# ---- UI: Clean and simple design ----
st.set_page_config(
    page_title="📘 AI Governance Chatbot", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Simple CSS
st.markdown(
    """
    <style>
    /* Clean, minimalist design */
    .main {
        padding: 2rem 1rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #4a5568;
        font-size: 1.1rem;
    }
    
    .chat-messages {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        height: 400px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    
    .message {
        max-width: 80%;
        padding: 12px 16px;
        border-radius: 18px;
        line-height: 1.4;
    }
    
    .user-message {
        background: #3182ce;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    
    .bot-message {
        background: white;
        color: #2d3748;
        border: 1px solid #e2e8f0;
        margin-right: auto;
        border-bottom-left-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .input-area {
        display: flex;
        gap: 10px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 24px;
        padding: 12px 20px;
    }
    
    .stButton button {
        border-radius: 50%;
        width: 50px;
        height: 50px;
        background: #3182ce;
        color: white;
        border: none;
    }
    
    .stButton button:hover {
        background: #2c5282;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        
        .title {
            font-size: 1.8rem;
        }
        
        .chat-messages {
            height: 350px;
        }
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <div class="header">
        <h1 class="title">📘 AI Governance Chatbot</h1>
        <p class="subtitle">Ask questions about AI governance in Nepali</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="message user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message bot-message">{message["content"]}</div>', unsafe_allow_html=True)

# Use a form to enable Enter key submission
with st.form(key='chat_form', clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input("", placeholder="Type your question in Nepali or English...", label_visibility="collapsed", key="input")
    with col2:
        submitted = st.form_submit_button("→")

# Handle user input when form is submitted (either by button click or Enter key)
if submitted and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get response from Gemini
    with st.spinner("Thinking..."):
        relevant_chunks = retrieve_relevant_chunks(user_input, chunks, embeddings)
        response = ask_gemini(user_input, relevant_chunks)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update the chat
    st.rerun()