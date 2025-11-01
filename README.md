
# 🐍 Python RAG Assistant

An intelligent assistant specialized in Python, powered by RAG (Retrieval-Augmented Generation) technology. Ask your Python questions and get accurate answers with code examples and documented references.

## ✨ Features

- 💬 **Conversational Interface**: Chat naturally with the assistant
- 📚 **Knowledge Base**: Access to comprehensive Python documentation
- 🔍 **Intelligent Search**: Retrieves relevant information from vector database
- 📝 **Code Examples**: Answers accompanied by practical examples
- 🎯 **Cited Sources**: Traceability of provided information
- ⚙️ **Configurable Parameters**: Adjust temperature and number of documents

## 🚀 How to Use

1. Type your Python question in the text field
2. Click "Send" or press Enter
3. The assistant searches its knowledge base
4. Receive a detailed answer with examples and sources

### Example Questions

- "How to create a list comprehension?"
- "What's the difference between a list and a tuple?"
- "How to handle exceptions in Python?"
- "Explain decorators to me"
- "How to read a CSV file?"

## 🛠️ Technologies Used

- **Streamlit**: User interface
- **LangChain**: Framework for LLM applications
- **ChromaDB**: Vector database
- **Sentence Transformers**: Embedding generation
- **Groq API**: Language model (GPT)

## ⚙️ Configuration

### Available Parameters

- **Temperature** (0.0 - 1.0): Controls response creativity
  - 0.0 = More precise and deterministic responses
  - 1.0 = More creative and varied responses
  
- **Top K documents** (1 - 10): Number of documents consulted
  - Higher numbers provide broader context
  - Recommended: 3-5 for good balance

## 📊 Architecture

```
User Question
    ↓
Convert to embedding
    ↓
Search in ChromaDB
    ↓
Retrieve relevant documents
    ↓
Generate response (LLM + context)
    ↓
Display with sources
```

## 🔒 Privacy

- No user data is stored permanently
- Conversations are not saved between sessions
- Use the "Clear History" button to delete the current session


## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Improve documentation

## 📧 Contact

For any questions or suggestions, feel free to open an issue on the GitHub repository.

---

Built with ❤️ using Streamlit and LangChain