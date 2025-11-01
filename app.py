import streamlit as st
from datetime import datetime
import time

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
    .source-box {
        background-color: #fff3e0;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        border-left: 3px solid #ff9800;
    }
    </style>
""", unsafe_allow_html=True)

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
    top_k = st.slider("Top K documents", 1, 10, 3)
    
    st.divider()
    
    # Statistiques
    st.subheader("📊 Statistics")
    st.metric("Questions", len([m for m in st.session_state.messages if m["role"] == "user"]))
    st.metric("Responses", len([m for m in st.session_state.messages if m["role"] == "assistant"]))
    
    st.divider()
    
    # Bouton pour effacer l'historique
    if st.button("🗑️ Delete", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_started = False
        st.rerun()
    
    st.divider()
    
    # Informations
    st.subheader("About")
    st.caption("RAG Assistant specialized in Python programming language")
    st.caption("Version 1.0.0")
    st.caption(f"Latest update: {datetime.now().strftime('%d/%m/%Y')}")

# En-tête principal
st.title("RAG Python Assistant ")
st.markdown("### Your python expert powered by IA")
st.markdown("---")

# Zone de bienvenue (affichée uniquement au début)
if not st.session_state.conversation_started:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**💡 Questions about Python syntax**")
    
    with col2:
        st.success("**📚 Documentation**")
    
    with col3:
        st.warning("**🔍 Bonnes pratiques**")

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
        
        # Afficher les sources si disponibles
        if "sources" in message and message["sources"]:
            with st.expander("📄 Sources "):
                for idx, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {idx}:** {source}")

# Zone de saisie utilisateur
st.markdown("---")
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Enter your question on Python... ",
        placeholder="Ex: What is list comprehension in Python ?",
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
    
    # Simulation de la réponse du RAG (à remplacer par votre logique)
    with st.spinner("🔍 Searching..."):
        time.sleep(1)  # Simuler le temps de traitement
        
        # REMPLACEZ CETTE PARTIE PAR VOTRE LOGIQUE RAG
        # Exemple de réponse simulée
        response = f"""ANswer about : "{user_input}"



Cette approche est recommandée car elle suit les conventions PEP 8."""
        
        # Sources simulées (remplacez par vos vraies sources)
        sources = [
            "Documentation Python 3.11 - Section Structures de données",
            "PEP 8 - Guide de style pour le code Python",
            "Python Tutorial - Chapitre 5"
        ]
    
    # Ajouter la réponse de l'assistant
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })
    
    # Recharger la page pour afficher les nouveaux messages
    st.rerun()

# Instructions en bas de page
if not st.session_state.conversation_started:
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>How to use this assistant ?</strong></p>
        <p>Ask your question and click on the button "send".</p>
        <p>The assistant will consult their knowledge base to provide you with an accurate and well-documented response.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    <p>Propulsed by RAG (Retrieval-Augmented Generation) | Specialized in Python 🐍</p>
</div>
""", unsafe_allow_html=True)