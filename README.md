# OCR Document Processing Pipeline

A comprehensive document processing application that extracts text, entities, and generates summaries from PDFs and images using enhanced Marker integration, Surya OCR, IndIC BERT, and Groq AI.

![IDP Architecture](./IDP_OCR.jpeg)


## Features

- **Multi-format Support**: PDF, JPG, JPEG, PNG, TIFF, BMP
- **Intelligent Processing**: Digital PDFs via Marker, scanned documents via Surya OCR
- **Multilingual Support**: Detects and translates Indian languages to English
- **Advanced Entity Extraction**: AI-powered extraction with Groq LLM
- **Interactive Visualization**: Hover-enabled bounding boxes with detailed metadata
- **MongoDB Storage**: Persistent storage of documents and extractions
- **Web Interface**: Multi-mode Streamlit GUI

##  Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BharatDBPG/India-AI-Intelligent-Document-Processing
   cd ocr
   ```

2. **Run setup script**:
   ```bash
   ./setup.sh
   ```

3. **Configure API keys**:
   ```bash
   # Update .streamlit/secrets.toml with your Groq API key
   GROQ_API_KEY = "your_key_here"
   ```

4. **Start the application**:
   ```bash
   ./start_app.sh
   ```

##  Usage

Access the web interface at `http://localhost:8501` and use the four main tabs:
- **Upload & Process**: Upload and process documents
- **Analysis Results**: View extracted entities and translations
- **Document Viewer**: Interactive document visualization with bounding boxes
- **Search Documents**: Search across processed documents

##  Requirements

- Python 3.8+
- MongoDB
- Groq API key

## 📄 License

MIT License

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.
