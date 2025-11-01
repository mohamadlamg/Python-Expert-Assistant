import streamlit as st
from datetime import datetime
import os
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from tqdm import tqdm

# Configuration de la page
st.set_page_config(
    page_title="Assistant Python RAG",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design professionnel
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Fonction pour charger les documents
def load_documents(documents_path) -> list:
    results = []
    for file in os.listdir(documents_path):
        if file.endswith('.txt'):
            file_path = os.path.join(documents_path, file)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                results.extend(loaded_docs)
            except Exception as e:
                st.warning(f"Error loading {file}: {str(e)}")
    
    content = []
    for doc in results:
        content.append(doc.page_content)
    
    return content

# Classe RAG Assistant
class RAGAssistant:
    """RAG assistant constructing"""
    def __init__(self, api_key):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=api_key
        )
        self.vector_db = VectorDB('rag_docs', "sentence-transformers/all-MiniLM-L6-v2") 
        self.prompt_template = ChatPromptTemplate.from_template("""
            You are a helpful, professional research assistant that answers questions about programming, web development, Python tkinter, French-English translation, etc.
            
            Use this following context to answer question: {context}   
            Question: {question}                                                         
            
            Guidelines:
            - Only answer questions based on the provided documents
            - If a question goes beyond scope, politely refuse: "I'm sorry I cannot help you about this subject"
            - If a question is unethical, illegal or unsafe, refuse to answer with the reason
            - Use clear, concise language with bullet points where appropriate
            - Do not show your documents titles or sources, just answer correctly the question
                                                                                                                
        """)
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def add_documents(self, documents):
        if not documents:
            return
        self.vector_db.add_documents(documents) 

    def search(self, query, n_results: int = 5):
        return self.vector_db.search(query, n_results)
    
    def ask(self, question: str, n_results: int = 5) -> str:
        search_results = self.search(question, n_results)
        
        if not search_results["documents"]:
            return "I'm sorry, I don't have information about this topic in my knowledge base."
        
        trimmed_docs = [doc[:500] for doc in search_results["documents"][:n_results]]
        context = "\n---\n".join(trimmed_docs)
        
        response = self.chain.invoke({
            "context": context,
            "question": question
        })
        
        return response

# Initialiser le RAG Assistant (une seule fois)
@st.cache_resource
def load_rag_system():
    """Charge le système RAG (mis en cache)"""
    try:
        # Récupérer la clé API depuis les secrets Streamlit
        api_key = st.secrets.get("GROQ_API_KEY", "")
        
        if not api_key:
            st.error("⚠️ GROQ_API_KEY not configured. Please add it in Streamlit secrets.")
            st.stop()
        
        # Initialiser l'assistant
        assistant = RAGAssistant(api_key)
        
        # Charger les documents
        if os.path.exists('data'):
            with st.spinner("Loading documents..."):
                documents = load_documents('data')
                if documents:
                    assistant.add_documents(documents)
                    st.success(f"✅ {len(documents)} documents loaded successfully!")
        else:
            st.warning("⚠️ 'data' folder not found. Please add your documents.")
        
        return assistant
    
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        st.stop()

# Charger le système RAG
assistant = load_rag_system()

# Initialisation de l'historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False

# Sidebar
with st.sidebar:
    st.image("https://www.python.org/static/community_logos/python-logo-generic.svg", width=200)
    st.title("⚙️ Configuration")
    
    # Paramètres du modèle
    st.subheader("RAG Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    top_k = st.slider("Top K documents", 1, 10, 5)
    
    st.divider()
    
    # Statistiques
    st.subheader("📊 Statistics")
    st.metric("Questions", len([m for m in st.session_state.messages if m["role"] == "user"]))
    st.metric("Responses", len([m for m in st.session_state.messages if m["role"] == "assistant"]))
    
    st.divider()
    
    # Bouton pour effacer l'historique
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_started = False
        st.rerun()
    
    st.divider()
    
    # Informations
    st.subheader("ℹ️ About")
    st.caption("RAG Assistant specialized in Python programming")
    st.caption("Version 1.0.0")
    st.caption(f"Latest update: {datetime.now().strftime('%d/%m/%Y')}")

# En-tête principal
st.title("🐍 RAG Python Assistant")
st.markdown("### Your Python expert powered by AI")
st.markdown("---")

# Zone de bienvenue (affichée uniquement au début)
if not st.session_state.conversation_started:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**💡 Questions about Python syntax**\n\nAsk questions about Python syntax, data structures, etc.")
    
    with col2:
        st.success("**📚 Documentation**\n\nGet detailed explanations with code examples.")
    
    with col3:
        st.warning("**🔍 Best Practices**\n\nDiscover Python best practices and conventions.")

# Affichage de l'historique des messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You</strong>
            <p style="margin-top: 0.5rem;">{message["content"]}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant</strong>
            <p style="margin-top: 0.5rem;">{message["content"]}</p>
        </div>
        """, unsafe_allow_html=True)

# Zone de saisie utilisateur
st.markdown("---")
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Enter your question about Python...",
        placeholder="Ex: What is list comprehension in Python?",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    send_button = st.button("Send 📤", use_container_width=True)

# Traitement de la question
if send_button and user_input:
    st.session_state.conversation_started = True
    
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Obtenir la réponse du RAG
    with st.spinner("🔍 Searching in knowledge base..."):
        try:
            response = assistant.ask(user_input, n_results=top_k)
        except Exception as e:
            response = f"Sorry, an error occurred: {str(e)}"
    
    # Ajouter la réponse de l'assistant
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    
    # Recharger la page pour afficher les nouveaux messages
    st.rerun()

# Instructions en bas de page
if not st.session_state.conversation_started:
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>How to use this assistant?</strong></p>
        <p>Ask your question in the field above and click "Send".</p>
        <p>The assistant will consult its knowledge base to provide you with an accurate and well-documented response.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    <p>Powered by RAG (Retrieval-Augmented Generation) | Specialized in Python 🐍</p>
</div>
""", unsafe_allow_html=True)