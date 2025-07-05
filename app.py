from flask import Flask, request, jsonify, render_template, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import PyPDF2
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma  # Using Chroma instead of FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.secret_key = 'pdf_summarizer_secret_key'

# Make sure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('chroma_db', exist_ok=True)  # Create folder for Chroma database

# Global variable to store the document vector store
pdf_vector_store = None
pdf_content = ""

# Initialize the LLM
llm = OllamaLLM(model="llama3.2")
output_parser = StrOutputParser()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def create_vector_store(text):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    embeddings = OllamaEmbeddings(model="llama3.2")
    
    # Create Chroma vector store instead of FAISS
    vector_store = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_store

# Route to serve the index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route to upload PDF
@app.route('/upload', methods=['POST'])
def upload_file():
    global pdf_vector_store, pdf_content
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text from PDF
        pdf_content = extract_text_from_pdf(file_path)
        
        # Create vector store for Q&A
        pdf_vector_store = create_vector_store(pdf_content)
        
        return jsonify({"message": "File uploaded successfully", "filename": filename})
    
    return jsonify({"error": "Invalid file type"}), 400

# Route to summarize PDF
@app.route('/summarize', methods=['GET'])
def summarize_pdf():
    global pdf_content
    
    if not pdf_content:
        return jsonify({"error": "No PDF uploaded or processed"}), 400
    
    # Initialize the prompt template for summarization
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional document summarizer. Create a concise summary of the following document content."),
        ("user", "Document content: {content}")
    ])
    
    # Create and invoke the summarization chain
    summarize_chain = summarize_prompt | llm | output_parser
    summary = summarize_chain.invoke({"content": pdf_content[:8000]})  # Limit length for LLM processing
    
    return jsonify({"summary": summary})

# Route for Q&A on PDF content
@app.route('/qa', methods=['POST'])
def qa_pdf():
    global pdf_vector_store
    
    if not pdf_vector_store:
        return jsonify({"error": "No PDF uploaded or processed"}), 400
    
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    # Search for relevant context in the document
    docs = pdf_vector_store.similarity_search(user_query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Initialize the prompt template for Q&A
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful PDF Q&A assistant. Answer the question based only on the provided context. If the answer is not in the context, say 'I don't have enough information to answer this question based on the uploaded PDF.'"),
        ("user", "Context: {context}\n\nQuestion: {query}")
    ])
    
    # Create and invoke the Q&A chain
    qa_chain = qa_prompt | llm | output_parser
    answer = qa_chain.invoke({"context": context, "query": user_query})
    
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)