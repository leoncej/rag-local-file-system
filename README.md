# rag-local-file-system
This is a local Retrieval-Augmented Generation (RAG) system designed to turn personal document libraries into a searchable, interactive intelligence base. This solution keeps all data on your machine rather than in the Cloud, utilizing Ollama (Llama 3) and HuggingFace Embeddings for a 100% private experience.
Key Features:
- Deep Document Intelligence: Specialized loaders for .pdf, .docx, .csv, .md, and .pptx.
- Scanned Document OCR: Integrated RapidOCR engine to read text from scanned certificates and images within PDFs.
- Live Folder Monitoring: A background Watchdog service that automatically detects file changes and re-indexes the knowledge base in real-time.
- Hybrid Model Architecture: Uses a lightweight "Librarian" model for high-speed searches and a powerful "Professor" model (Llama 3) for reasoning.
- Interactive UI: A professional Streamlit dashboard with asynchronous progress tracking and source transparency.

Technical Stack: LangChain: LLM Orchestration, Ollama (Llama3): Local LLM, HuggingFace (all-MiniLM-L6-v2): Embeddings, ChromaDB: Vector Database, Streamlit: Frontend, Python Watchdog: File Monitoring, RapidOCR: OCR Engine
