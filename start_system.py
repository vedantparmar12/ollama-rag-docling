#!/usr/bin/env python3
"""
RAG System Launcher for Windows
================================
This script properly sets up the Python path and starts all services.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

# Set up project root and Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)

# Add project root to Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set environment variable for subprocesses
os.environ['PYTHONPATH'] = str(PROJECT_ROOT)

# Load .env file
from dotenv import load_dotenv
load_dotenv()

def print_header(message):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {message}")
    print("="*70)

def check_ollama():
    """Check if Ollama is running."""
    print("\n[1/4] Checking Ollama status...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
    except requests.exceptions.RequestException:
        pass

    print("❌ ERROR: Ollama is not running!")
    print("Please start Ollama first:")
    print("  1. Open a new terminal")
    print("  2. Run: ollama serve")
    return False

def check_models():
    """Check if required models are installed."""
    print("\n[2/4] Checking required models...")

    required_models = ["llama3.2", "smollm2", "nomic-embed-text"]

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        installed_models = result.stdout.lower()

        for model in required_models:
            if model in installed_models:
                print(f"  ✅ {model}")
            else:
                print(f"  ⚠️  {model} not found - pulling...")
                subprocess.run(["ollama", "pull", model], check=True)
                print(f"  ✅ {model} installed")

        print("✅ All required models are available")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error checking models: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n[3/4] Setting up directories...")

    directories = [
        "lancedb",
        "shared_uploads",
        "logs",
        "index_store",
        "index_store/overviews",
        "index_store/bm25",
        "index_store/graph"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ Directories created")

def start_services():
    """Start all services."""
    print("\n[4/4] Starting services...")

    # Start RAG API Server
    print("🔧 Starting RAG API Server on port 8001...")
    rag_api_cmd = [sys.executable, "rag_system/api_server.py"]
    subprocess.Popen(
        rag_api_cmd,
        env=os.environ.copy(),
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    )
    time.sleep(3)

    # Start Backend Server
    print("🔧 Starting Backend Server on port 8000...")
    backend_cmd = [sys.executable, "server.py"]
    subprocess.Popen(
        backend_cmd,
        cwd=str(PROJECT_ROOT / "backend"),
        env=os.environ.copy(),
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    )
    time.sleep(3)

    # Start Frontend Server
    print("🔧 Starting Frontend Server on port 3000...")
    # On Windows, use npm.cmd instead of npm
    npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
    frontend_cmd = [npm_cmd, "run", "dev"]
    subprocess.Popen(
        frontend_cmd,
        env=os.environ.copy(),
        shell=True,  # Use shell on Windows for npm
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    )
    time.sleep(3)

def main():
    """Main launcher function."""
    print_header("🚀 Starting RAG System - Ollama Edition")

    # Check prerequisites
    if not check_ollama():
        input("\nPress Enter to exit...")
        return 1

    if not check_models():
        input("\nPress Enter to exit...")
        return 1

    # Setup
    create_directories()

    # Start services
    start_services()

    # Print success message
    print_header("✅ RAG System Started Successfully!")
    print("\n🌐 Access your application at:")
    print("   Frontend:   http://localhost:3000")
    print("   Backend:    http://localhost:8000")
    print("   RAG API:    http://localhost:8001")
    print("   Ollama:     http://localhost:11434")
    print("\n📊 Models in use:")
    print("   Generation: llama3.2:latest")
    print("   Enrichment: smollm2:latest")
    print("   Embedding:  nomic-embed-text:latest")
    print("\n📝 Three new windows have opened for each service.")
    print("   Close those windows to stop the services.")
    print("\n🎯 Next steps:")
    print("   1. Open http://localhost:3000 in your browser")
    print("   2. Create a new chat session")
    print("   3. Upload some PDF documents")
    print("   4. Start asking questions!")
    print("\n" + "="*70)

    # Open browser
    print("\nOpening browser in 3 seconds...")
    time.sleep(3)
    import webbrowser
    webbrowser.open("http://localhost:3000")

    print("\nSystem is running. Close service windows when done.")
    print("="*70)

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
