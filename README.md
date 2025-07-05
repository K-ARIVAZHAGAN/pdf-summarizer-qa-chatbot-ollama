# PDF Summarizer & Q&A Chatbot with Ollama

A Flask-based web application that allows users to upload PDF documents and interact with them through AI-powered summarization and question-answering capabilities using LangChain and Ollama.

## Features

- **PDF Upload**: Upload PDF documents (up to 16MB)
- **AI-Powered Summarization**: Generate intelligent summaries of uploaded PDFs using Ollama
- **Interactive Q&A Chat**: Ask questions about the PDF content and get AI-powered responses
- **Vector Search**: Uses Chroma vector database for efficient document retrieval
- **Web Interface**: Clean, user-friendly web interface with Bootstrap styling

## Technology Stack

- **Backend**: Flask (Python)
- **AI/ML**: 
  - LangChain for document processing and chaining
  - Ollama with Llama 3.2 model for text generation
  - Chroma vector database for document embeddings
- **Frontend**: HTML with Bootstrap styling
- **File Processing**: PyPDF2 for PDF text extraction

## Prerequisites

Before running the application, ensure you have:

1. **Python 3.7+** installed
2. **Ollama** installed and running with the `llama3.2` model
3. Required Python packages (see Installation section)

## Installation

1. **Clone or download the project**
   ```bash
   cd Q&A_FINAL/Q&A
   ```

2. **Install required dependencies**
   ```bash
   pip install flask flask-cors werkzeug PyPDF2 langchain-core langchain-ollama langchain-community langchain-text-splitters chromadb
   ```

3. **Install and setup Ollama**
   ```bash
   # Install Ollama (visit https://ollama.ai for platform-specific instructions)
   # Pull the required model
   ollama pull llama3.2
   ```

4. **Ensure Ollama is running**
   ```bash
   ollama serve
   ```

## Project Structure

```
Q&A/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Directory for uploaded PDFs
├── chroma_db/           # Chroma vector database storage
└── DEMO_OUTPUT/         # Screenshots of application demo
    ├── Screenshot 2025-07-05 200235.png
    ├── Screenshot 2025-07-05 200318.png
    ├── Screenshot 2025-07-05 200327.png
    └── Screenshot 2025-07-05 200345.png
```

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to `http://localhost:5000`

3. **Upload a PDF**
   - Click "Choose File" and select a PDF document
   - Click "Upload" to process the document

4. **Get a Summary**
   - After uploading, click "Summarize" to get an AI-generated summary

5. **Ask Questions**
   - Type your question in the text area
   - Click "Ask Question" to get AI-powered answers based on the PDF content

## API Endpoints

### Upload PDF
- **POST** `/upload`
- Upload a PDF file for processing
- Returns: Upload confirmation with filename

### Summarize Document
- **GET** `/summarize`
- Generate a summary of the uploaded PDF
- Returns: AI-generated summary

### Question & Answer
- **POST** `/qa`
- Ask questions about the PDF content
- Body: `{"query": "Your question here"}`
- Returns: AI-generated answer based on document context

## Demo Output

The application interface and functionality are demonstrated in the screenshots located in the `DEMO_OUTPUT/` folder:

1. **Screenshot 2025-07-05 200235.png** - Main application interface
2. **Screenshot 2025-07-05 200318.png** - PDF upload functionality
3. **Screenshot 2025-07-05 200327.png** - Document summarization feature
4. **Screenshot 2025-07-05 200345.png** - Q&A interaction example

## Configuration

The application uses the following default configurations:

- **Upload folder**: `uploads/`
- **Allowed file types**: PDF only
- **Maximum file size**: 16MB
- **Vector database**: Chroma (stored in `chroma_db/`)
- **Text chunk size**: 1000 characters with 200 character overlap
- **LLM model**: Llama 3.2 via Ollama

## Error Handling

The application includes comprehensive error handling for:

- Invalid file types
- Missing files
- Large file uploads
- Missing PDF content
- Invalid queries
- LLM processing errors

## Troubleshooting

### Common Issues

1. **"No module named 'langchain'"**
   ```bash
   pip install langchain-core langchain-ollama langchain-community
   ```

2. **"Ollama model not found"**
   ```bash
   ollama pull llama3.2
   ```

3. **"Connection error to Ollama"**
   - Ensure Ollama service is running: `ollama serve`
   - Check if the model is available: `ollama list`

4. **"File upload fails"**
   - Check file size (must be under 16MB)
   - Ensure file is in PDF format
   - Verify upload directory permissions

## Security Considerations

- File uploads are restricted to PDF format only
- Filenames are secured using `werkzeug.utils.secure_filename`
- File size is limited to 16MB
- Input validation is performed on all user inputs

## Performance Notes

- Large PDFs may take longer to process
- Vector similarity search is optimized for quick retrieval
- Document chunks are limited to 8000 characters for LLM processing
- Chroma database persists embeddings for faster subsequent queries

## License

This project is created for educational and demonstration purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
