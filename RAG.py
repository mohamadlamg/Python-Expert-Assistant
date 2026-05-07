import os 
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from Vertical_AI.vectordb import VectorDB
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv('GROQ_API_KEY')


def load_documents(documents_path)-> list:
    results = []
    for file in os.listdir(documents_path):
        if file.endswith('.txt'):    #Can add others extensions
            file_path = os.path.join(documents_path,file)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                results.extend(loaded_docs)
                print(f"Successfully load : {file}")
            
            except FileNotFoundError :
                print("Sorry, no files found")

            except Exception as e:
                print(f"Error during loading.. : {file}:{str(e)}")
    
    print(f"total documents loaded : {len(results)}")

    content = []
    for _,doc in tqdm(enumerate(results)):
       content.append(doc.page_content)

    print(f"{len(content)} datas loaded")
    
    return content

llm = ChatGroq(model="llama-3.1-8b-instant",
               temperature=0.7,
               api_key=api_key)


class RAGAssistant:
    """RAG assistant constructing"""
    def __init__(self):
       self.llm = llm
       self.vector_db = VectorDB('rag_docs',"sentence-transformers/all-MiniLM-L6-v2") 
       self.prompt_template = ChatPromptTemplate.from_template("""
    You are a Python-only expert assistant.
    
    STEP 1 - Check the question:
    Is the question "{question}" directly asking about Python programming? 
    If NO → respond ONLY with: "I'm sorry, I can only help with Python-related questions." STOP here.
    If YES → continue to STEP 2.
    
    STEP 2 - Answer using context:
    {context}
    
    Guidelines:
    - Only answer based on the provided documents
    - Use clear, concise language with bullet points where appropriate
    - Do not show document titles or sources
    - If unethical or unsafe, refuse with the reason
""")
       self.chain = self.prompt_template | self.llm | StrOutputParser()
       print("RAG components successfully installed")

      

    def add_documents(self,documents):

        if not documents:
            print("Not document found")
            return

        
        print(f"Processing {len(documents)} documents")
        self.vector_db.add_documents(documents) 

    def search(self,query,n_results:int=5):
       
       
        return self.vector_db.search(query, n_results)
    def ask(self, question: str) -> str:
        
        search_results = self.search(question)
        
        if not search_results["documents"]:
            return "I'm sorry, I don't have information about this topic in my knowledge base."
        
        
        trimmed_docs = [doc[:500] for doc in search_results["documents"][:2]]
        context = "\n---\n".join(trimmed_docs)
        
        response = self.chain.invoke({
            "context": context,
            "question": question
        })
   
        
        return response



def main ():
    try:
        print("Initialization...")
        assistant = RAGAssistant()

        print("Data loading..")
        sample_docs = load_documents('data')
        print(f"{len(sample_docs)} documents loaded")
        assistant.add_documents(sample_docs)
        print("RAG assistant ready !")

        while True :
            question = input("Enter a question or 'quit' to exit : ")
            if question.lower() == 'quit' :
                break
            else :
                print("\nYour answer will be ready soon...")
                response = assistant.ask(question)
                print(response)
    except Exception as e :
        print(f"Something went wrong...{e}")

if __name__ == '__main__':
    main()