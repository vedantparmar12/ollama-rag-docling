# 🚀 Windows Setup Guide - RAG System with Ollama

## ✅ Setup Complete!

Your RAG system is now configured to run on Windows with Ollama.

## 📦 What's Been Configured

### Models in Use:
- **Generation Model**: `llama3.2:latest` (Main LLM for answers)
- **Enrichment Model**: `smollm2:latest` (Lightweight routing/enrichment)
- **Embedding Model**: `nomic-embed-text:latest` (Document embeddings)

### Services:
- **Frontend**: http://localhost:3000 (Next.js UI)
- **Backend**: http://localhost:8000 (FastAPI server)
- **RAG API**: http://localhost:8001 (RAG processing)
- **Ollama**: http://localhost:11434 (LLM inference)

## 🎮 Quick Commands

### Start the System (Recommended)
```bash
python start_system.py
```
**OR** double-click `run_simple.bat`

This will:
1. Check if Ollama is running
2. Verify required models are installed
3. Create necessary directories
4. Start all three services in separate windows
5. Open your browser to http://localhost:3000

### Alternative: Original Batch Script
```bash
run.bat
```
(Same functionality, but pure batch script)

### Stop the System
```bash
stop.bat
```
Stops all RAG system services (Frontend, Backend, RAG API)

### Check Status
```bash
status.bat
```
Shows which services are running and their health status

## 📋 Prerequisites

Make sure you have:
- ✅ Python 3.8+ installed
- ✅ Node.js 14+ installed
- ✅ Ollama installed and running
- ✅ All dependencies installed (already done via uv venv and npm)

## 🔧 First-Time Setup

### 1. Ensure Ollama is Running
```bash
# Start Ollama if not running
ollama serve
```

### 2. Pull Required Models (if not already done)
```bash
ollama pull llama3.2
ollama pull smollm2
ollama pull nomic-embed-text
```

### 3. Run the System
```bash
run.bat
```

## 📚 Usage Guide

### Step 1: Start the System
Double-click `run.bat` or run it from command prompt

### Step 2: Access the UI
Your browser will automatically open to http://localhost:3000

### Step 3: Create a Chat Session
Click "New Chat" in the UI

### Step 4: Upload Documents
- Click the upload button
- Select PDF files you want to query
- Wait for indexing to complete

### Step 5: Ask Questions
Type your questions about the uploaded documents!

## 🐛 Troubleshooting

### Ollama Not Running
**Error**: `Ollama is not running on port 11434`

**Solution**:
```bash
# Start Ollama
ollama serve

# In another terminal, verify it's running
curl http://localhost:11434/api/tags
```

### Port Already in Use
**Error**: `Port 3000/8000/8001 already in use`

**Solution**:
```bash
# Stop all services
stop.bat

# Check what's using the port
netstat -ano | findstr :3000

# Kill the process (replace PID with actual process ID)
taskkill /F /PID <PID>
```

### Models Not Found
**Error**: `Model llama3.2 not found`

**Solution**:
```bash
# Pull the required models
ollama pull llama3.2
ollama pull smollm2
ollama pull nomic-embed-text

# Verify models are installed
ollama list
```

### Python Import Errors
**Error**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
```bash
# Activate your virtual environment
# Then reinstall dependencies
pip install -r requirements.txt
```

### Node.js/NPM Errors
**Error**: `Cannot find module 'xxx'`

**Solution**:
```bash
# Reinstall node dependencies
npm install

# If issues persist, clear cache
npm cache clean --force
npm install
```

## 📁 Directory Structure

```
localGPT-main/
├── .env                     # Environment configuration
├── run.bat                  # Start all services
├── stop.bat                 # Stop all services
├── status.bat               # Check service status
├── frontend/                # Next.js frontend
├── backend/                 # FastAPI backend
├── rag_system/             # RAG processing engine
├── lancedb/                # Vector database storage
├── shared_uploads/         # Uploaded documents
├── index_store/            # Document indices
└── logs/                   # Application logs
```

## 🔍 Service Details

### Frontend (Port 3000)
- Next.js-based web interface
- Real-time chat with streaming responses
- Document upload and management
- Session history

### Backend (Port 8000)
- FastAPI server
- Chat history management
- Document processing coordination
- SQLite database for chat sessions

### RAG API (Port 8001)
- Document indexing pipeline
- Vector search and retrieval
- LLM-powered answer generation
- Reranking and context optimization

### Ollama (Port 11434)
- Local LLM inference
- Model management
- GPU acceleration (if available)

## ⚙️ Configuration

Edit `.env` file to customize:

```env
# Change models
DEFAULT_GENERATION_MODEL=llama3.2:latest
DEFAULT_ENRICHMENT_MODEL=smollm2:latest

# Adjust performance
MAX_CONCURRENT_REQUESTS=5
EMBEDDING_BATCH_SIZE=32
MAX_CONTEXT_LENGTH=4096

# Enable advanced features
ENABLE_CONTEXT_PRUNING=true
ENABLE_MULTIMODAL_EMBEDDINGS=false
```

## 🚀 Advanced Usage

### Using Different Models
1. Pull a different model:
   ```bash
   ollama pull qwen2:7b
   ```

2. Update `.env`:
   ```env
   DEFAULT_GENERATION_MODEL=qwen2:7b
   ```

3. Restart the system:
   ```bash
   stop.bat
   run.bat
   ```

### Enabling Advanced Features
Edit `.env` to enable features:
```env
ENABLE_SELF_CONSISTENCY=true      # Better accuracy, slower
ENABLE_MULTIMODAL_EMBEDDINGS=true # Better for code/tables
ENABLE_CONTEXT_PRUNING=true       # Reduce token usage
```

## 📊 Performance Tips

1. **Use GPU acceleration**: Ollama automatically uses GPU if available
2. **Adjust batch sizes**: Lower `EMBEDDING_BATCH_SIZE` if running out of memory
3. **Reduce context window**: Lower `MAX_CONTEXT_LENGTH` for faster responses
4. **Use smaller models**: Switch to smaller models for faster inference

## 🔐 Security Notes

For development:
- API key authentication is disabled
- CORS is open to localhost
- Rate limiting is permissive

For production, edit `.env`:
```env
API_KEY_REQUIRED=true
CORS_ORIGINS=https://yourdomain.com
RATE_LIMIT_REQUESTS=10
```

## 📞 Support

- **Documentation**: Check README.md for full details
- **Issues**: Create an issue on GitHub
- **Advanced Features**: See ADVANCED_FEATURES_2025.md
- **Docker**: See DOCKER_README.md for Docker setup

## ✨ Next Steps

1. ✅ System is configured and ready
2. 🚀 Run `run.bat` to start
3. 📝 Upload some PDFs
4. 💬 Start asking questions!
5. 🔧 Customize settings in `.env` as needed

---

**Happy RAG-ing! 🎉**
