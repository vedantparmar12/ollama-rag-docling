@echo off
REM Simple Index Creation Script for LocalGPT RAG System (Windows)
REM Usage: simple_create_index.bat "Index Name" "path\to\document.pdf" [additional_files...]

setlocal enabledelayedexpansion

if "%~1"=="" goto show_usage
if "%~2"=="" goto show_usage

set "index_name=%~1"
shift

REM Check prerequisites
echo [INFO] Checking prerequisites...

REM Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3 is required but not installed.
    exit /b 1
)

REM Check if we're in the right directory
if not exist "run_system.py" (
    echo [ERROR] This script must be run from the LocalGPT project root directory.
    exit /b 1
)
if not exist "rag_system" (
    echo [ERROR] This script must be run from the LocalGPT project root directory.
    exit /b 1
)

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ollama is not running. Please start Ollama first:
    echo   ollama serve
    exit /b 1
)

echo [SUCCESS] Prerequisites check passed

REM Collect and validate documents
set valid_count=0
set "valid_docs="

:collect_docs
if "%~1"=="" goto create_index_now

set "doc=%~1"
if exist "!doc!" (
    REM Check file extension
    set "ext=%~x1"
    if /i "!ext!"==".pdf" set /a valid_count+=1 & set "valid_docs=!valid_docs! "!doc!""
    if /i "!ext!"==".txt" set /a valid_count+=1 & set "valid_docs=!valid_docs! "!doc!""
    if /i "!ext!"==".docx" set /a valid_count+=1 & set "valid_docs=!valid_docs! "!doc!""
    if /i "!ext!"==".md" set /a valid_count+=1 & set "valid_docs=!valid_docs! "!doc!""
    if /i "!ext!"==".html" set /a valid_count+=1 & set "valid_docs=!valid_docs! "!doc!""
    if /i "!ext!"==".htm" set /a valid_count+=1 & set "valid_docs=!valid_docs! "!doc!""

    if /i "!ext!"==".pdf" echo [INFO] ✓ Valid document: !doc!
    if /i "!ext!"==".txt" echo [INFO] ✓ Valid document: !doc!
    if /i "!ext!"==".docx" echo [INFO] ✓ Valid document: !doc!
    if /i "!ext!"==".md" echo [INFO] ✓ Valid document: !doc!
    if /i "!ext!"==".html" echo [INFO] ✓ Valid document: !doc!
    if /i "!ext!"==".htm" echo [INFO] ✓ Valid document: !doc!
) else (
    echo [WARNING] File not found: !doc! (skipping^)
)

shift
goto collect_docs

:create_index_now
if %valid_count% equ 0 (
    echo [ERROR] No valid documents found.
    exit /b 1
)

echo [INFO] Creating index: %index_name%
echo [INFO] Processing %valid_count% document(s^)

REM Create temporary Python script
set "temp_script=%temp%\create_index_temp_%random%.py"

(
    echo import sys
    echo import os
    echo import json
    echo sys.path.insert(0, os.getcwd(^)^)
    echo.
    echo from rag_system.main import PIPELINE_CONFIGS
    echo from rag_system.pipelines.indexing_pipeline import IndexingPipeline
    echo from rag_system.utils.ollama_client import OllamaClient
    echo from backend.database import ChatDatabase
    echo import uuid
    echo.
    echo def create_index_simple(^):
    echo     try:
    echo         # Initialize database
    echo         db = ChatDatabase(^)
    echo.
    echo         # Create index record
    echo         index_id = db.create_index(
    echo             name="%index_name%",
    echo             description="Created with simple_create_index.bat",
    echo             metadata={
    echo                 "chunk_size": 512,
    echo                 "chunk_overlap": 64,
    echo                 "enable_enrich": True,
    echo                 "enable_latechunk": True,
    echo                 "retrieval_mode": "hybrid",
    echo                 "created_by": "simple_create_index.bat"
    echo             }
    echo         ^)
    echo.
    echo         # Add documents to index
    echo         documents = [!valid_docs!]
    echo         for doc_path in documents:
    echo             if doc_path.strip(^):
    echo                 filename = os.path.basename(doc_path.strip(^)^)
    echo                 db.add_document_to_index(index_id, filename, os.path.abspath(doc_path.strip(^)^)^)
    echo.
    echo         # Initialize pipeline
    echo         config = PIPELINE_CONFIGS.get("default", {}^)
    echo         ollama_client = OllamaClient(^)
    echo         ollama_config = {
    echo             "generation_model": "qwen3:0.6b",
    echo             "embedding_model": "qwen3:0.6b"
    echo         }
    echo.
    echo         pipeline = IndexingPipeline(config, ollama_client, ollama_config^)
    echo.
    echo         # Process documents
    echo         valid_docs_list = [doc.strip(^) for doc in documents if doc.strip(^) and os.path.exists(doc.strip(^)^)]
    echo         if valid_docs_list:
    echo             pipeline.process_documents(valid_docs_list^)
    echo.
    echo         print(f"✅ Index '{%index_name%}' created successfully!"^)
    echo         print(f"Index ID: {index_id}"^)
    echo         print(f"Processed {len(valid_docs_list^)} documents"^)
    echo.
    echo         return index_id
    echo.
    echo     except Exception as e:
    echo         print(f"❌ Error creating index: {e}"^)
    echo         import traceback
    echo         traceback.print_exc(^)
    echo         return None
    echo.
    echo if __name__ == "__main__":
    echo     create_index_simple(^)
) > "%temp_script%"

REM Run the Python script
python "%temp_script%"

REM Clean up
del "%temp_script%" 2>nul

echo [SUCCESS] Index creation completed!
echo [INFO] You can now use the index in the LocalGPT interface.

goto :eof

:show_usage
echo Usage: %~nx0 "Index Name" "path\to\document.pdf" [additional_files...]
echo.
echo Examples:
echo   %~nx0 "My Documents" "document.pdf"
echo   %~nx0 "Research Papers" "paper1.pdf" "paper2.pdf" "notes.txt"
echo   %~nx0 "Invoice Collection" invoices\*.pdf
echo.
echo Supported file types: PDF, TXT, DOCX, MD, HTML
goto :eof

endlocal
