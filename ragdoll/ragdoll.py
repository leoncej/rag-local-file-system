# The Backend Engine
import os
import shutil
import time
from threading import Thread

# --- Document Processing ---
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader, CSVLoader,
    UnstructuredPowerPointLoader, UnstructuredHTMLLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Vector Database & AI ---
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# --- Watchdog for Live Updates ---
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class RAGdollHandler(FileSystemEventHandler):
    """
    The 'Security Guard': Listens for file changes and tells RAGdoll to update.
    """
    def __init__(self, rag_instance, paths):
        self.rag = rag_instance
        self.paths = paths
        self.last_sync = 0

    def on_modified(self, event):
        # Prevent 'Sync Loops' by waiting 5 seconds between updates
        if time.time() - self.last_sync > 5:
            if not event.is_directory and event.src_path.endswith(('.pdf', '.txt', '.docx', '.md', '.csv', '.pptx', '.html')):
                print(f"🔄 Watchdog: Change detected in {event.src_path}. Updating...")
                self.rag.ingest_docs(self.paths)
                self.last_sync = time.time()

class RAGdoll:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self.observer = None

        # THE LIBRARIAN: Fast embeddings model for indexing
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # THE PROFESSOR: Large Language Model for chatting
        self.llm = ChatOllama(model="llama3", temperature=0.1)

        # SHORT-TERM MEMORY: Remembers the last few chat turns
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.vector_store = None

    def start_watchdog(self, paths):
        """Starts background monitoring of folders."""
        if self.observer:
            self.observer.stop()

        event_handler = RAGdollHandler(self, paths)
        self.observer = Observer()
        for path in paths:
            clean_path = os.path.abspath(os.path.expanduser(path.strip()))
            if os.path.exists(clean_path):
                self.observer.schedule(event_handler, clean_path, recursive=True)
        self.observer.start()

    def ingest_docs(self, paths_to_scan, progress_callback=None):
        """Wipes the DB and rebuilds it with all files in the paths."""
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

        # Mapping extensions to their specific loaders
        loader_map = {
            ".txt": TextLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".md": UnstructuredMarkdownLoader,
            ".csv": CSVLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".html": UnstructuredHTMLLoader
        }

        documents = []
        all_files = []

        # 1. Discovery (Expanded Filter)
        for path in paths_to_scan:
            clean_path = os.path.abspath(os.path.expanduser(path.strip()))
            for root, _, files in os.walk(clean_path):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if (ext in loader_map or ext == ".pdf") and not f.startswith('.'):
                        all_files.append(os.path.join(root, f))

        # 2. Loading with Logic
        for i, file_path in enumerate(all_files):
            ext = os.path.splitext(file_path)[1].lower()
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(file_path, extract_images=True)
                else:
                    loader_class = loader_map.get(ext)
                    loader = loader_class(file_path)

                documents.extend(loader.load())

                if progress_callback:
                    prog = (i + 1) / len(all_files) * 0.9
                    progress_callback(prog, f"Reading: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Could not read {file_path}: {e}")

        # Step 3: Indexing
        if not documents: return "No docs found."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        if progress_callback: progress_callback(0.95, "🧠 Thinking...")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        return f"🚀 Indexed {len(documents)} files."

    def get_chat_response(self, user_query):
        """Retrieves info and generates an answer."""
        if not self.vector_store:
            if os.path.exists(self.db_path):
                self.vector_store = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
            else:
                return {"answer": "Sync first!", "source_documents": []}

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            return_source_documents=True
        )
        return chain.invoke({"question": user_query})
