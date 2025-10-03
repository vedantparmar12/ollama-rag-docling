"""
Docling Integration Module - Seamless drop-in replacement for existing pipeline
Preserves your custom reranker and chunking while adding Docling enhancements
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os

# Import your existing components
from .chunking import MarkdownRecursiveChunker
from .document_converter import DocumentConverter

# Import new Docling enhancements
try:
    from .docling_vlm_converter import DoclingVLMConverter
    from .audio_transcriber import AudioTranscriber
    from .enhanced_hybrid_chunker import EnhancedHybridChunker
    DOCLING_ENHANCEMENTS_AVAILABLE = True
except ImportError:
    DOCLING_ENHANCEMENTS_AVAILABLE = False
    print("⚠️ Docling enhancements not available")


class DoclingIntegratedPipeline:
    """
    Enhanced document processing pipeline with Docling features.

    Features Added:
    - ✅ VLM-powered document understanding (tables, formulas, code)
    - ✅ Audio transcription (Whisper)
    - ✅ Enhanced hybrid chunking (preserves your logic)
    - ✅ Multi-format support (PPTX, XLSX, MP3, etc.)

    Preserved:
    - ✅ Your custom MarkdownRecursiveChunker
    - ✅ Your QwenReranker
    - ✅ Your LanceDB + graph extraction
    - ✅ Your Ollama-based LLM infrastructure
    """

    # Extended format support
    SUPPORTED_FORMATS = {
        # Documents (original + enhanced)
        '.pdf': 'document',
        '.docx': 'document',
        '.doc': 'document',
        '.pptx': 'document',  # NEW
        '.ppt': 'document',   # NEW
        '.xlsx': 'document',  # NEW
        '.xls': 'document',   # NEW
        '.html': 'document',
        '.htm': 'document',
        '.md': 'document',
        '.txt': 'document',
        # Audio (NEW)
        '.mp3': 'audio',
        '.wav': 'audio',
        '.m4a': 'audio',
        '.flac': 'audio',
    }

    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 200,
        tokenizer_model: str = "Qwen/Qwen3-Embedding-0.6B",
        use_docling_vlm: bool = True,
        use_docling_chunking: bool = True,
        enable_audio: bool = True,
        extract_tables: bool = True,
        extract_formulas: bool = True,
        extract_code: bool = True
    ):
        """
        Initialize integrated pipeline.

        Args:
            max_chunk_size: Max tokens per chunk
            min_chunk_size: Min tokens per chunk
            tokenizer_model: Tokenizer for counting
            use_docling_vlm: Use Granite-Docling VLM (recommended)
            use_docling_chunking: Use hybrid chunking (recommended)
            enable_audio: Enable audio transcription
            extract_tables: Extract table structures
            extract_formulas: Extract formulas
            extract_code: Extract code blocks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.use_docling_vlm = use_docling_vlm
        self.use_docling_chunking = use_docling_chunking
        self.enable_audio = enable_audio

        # Initialize document converter (fallback to original)
        self.document_converter = DocumentConverter()

        # Initialize VLM converter if enabled
        self.vlm_converter = None
        if use_docling_vlm and DOCLING_ENHANCEMENTS_AVAILABLE:
            try:
                self.vlm_converter = DoclingVLMConverter(
                    use_vlm=True,
                    extract_tables=extract_tables,
                    extract_formulas=extract_formulas,
                    extract_code=extract_code
                )
                print("✅ VLM converter enabled")
            except Exception as e:
                print(f"⚠️ VLM converter failed: {e}. Using standard converter.")

        # Initialize audio transcriber if enabled
        self.audio_transcriber = None
        if enable_audio and DOCLING_ENHANCEMENTS_AVAILABLE:
            try:
                self.audio_transcriber = AudioTranscriber(
                    model="turbo",
                    include_timestamps=True
                )
                print("✅ Audio transcriber enabled")
            except Exception as e:
                print(f"⚠️ Audio transcriber failed: {e}")

        # Initialize chunker (enhanced or original)
        if use_docling_chunking and DOCLING_ENHANCEMENTS_AVAILABLE:
            try:
                self.chunker = EnhancedHybridChunker(
                    max_chunk_size=max_chunk_size,
                    min_chunk_size=min_chunk_size,
                    tokenizer_model=tokenizer_model,
                    use_docling_structure=True,
                    preserve_tables=extract_tables,
                    preserve_code=extract_code,
                    preserve_formulas=extract_formulas
                )
                print("✅ Enhanced hybrid chunker enabled")
            except Exception as e:
                print(f"⚠️ Enhanced chunker failed: {e}. Using original.")
                self.chunker = MarkdownRecursiveChunker(
                    max_chunk_size=max_chunk_size,
                    min_chunk_size=min_chunk_size,
                    tokenizer_model=tokenizer_model
                )
        else:
            self.chunker = MarkdownRecursiveChunker(
                max_chunk_size=max_chunk_size,
                min_chunk_size=min_chunk_size,
                tokenizer_model=tokenizer_model
            )

        print(f"\n📊 Pipeline Initialized:")
        print(f"   VLM Converter: {'✅ Enabled' if self.vlm_converter else '❌ Disabled'}")
        print(f"   Audio Transcriber: {'✅ Enabled' if self.audio_transcriber else '❌ Disabled'}")
        print(f"   Enhanced Chunker: {'✅ Enabled' if use_docling_chunking else '❌ Using Original'}")
        print(f"   Supported Formats: {len(self.SUPPORTED_FORMATS)}")

    def process_document(
        self,
        file_path: str,
        document_id: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process document through enhanced pipeline.

        Args:
            file_path: Path to document
            document_id: Document identifier
            document_metadata: Additional metadata

        Returns:
            Tuple of (chunks, enriched_metadata)
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        # Validate format
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {file_ext}")

        format_type = self.SUPPORTED_FORMATS[file_ext]

        # Route to appropriate processor
        if format_type == 'audio':
            return self._process_audio(file_path, document_id, document_metadata)
        else:
            return self._process_document(file_path, document_id, document_metadata)

    def _process_audio(
        self,
        file_path: Path,
        document_id: str,
        document_metadata: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process audio file with transcription."""
        if not self.audio_transcriber:
            raise RuntimeError("Audio transcription not enabled")

        print(f"🎙️ Processing audio: {file_path.name}")

        # Transcribe audio
        transcript, audio_metadata = self.audio_transcriber.transcribe(str(file_path))

        # Merge metadata
        merged_metadata = {**(document_metadata or {}), **audio_metadata}

        # Chunk transcript
        chunks = self.chunker.chunk(
            text=transcript,
            document_id=document_id,
            document_metadata=merged_metadata
        )

        print(f"✅ Audio processed: {len(chunks)} chunks from transcript")

        return chunks, merged_metadata

    def _process_document(
        self,
        file_path: Path,
        document_id: str,
        document_metadata: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process document with VLM or standard converter."""
        print(f"📄 Processing document: {file_path.name}")

        # Convert to markdown
        docling_doc = None
        if self.vlm_converter:
            # Use VLM converter for enhanced extraction
            conversion_results = self.vlm_converter.convert_to_markdown(str(file_path))
            if conversion_results:
                markdown_text, doc_metadata, docling_doc = conversion_results[0]
            else:
                raise RuntimeError(f"Failed to convert {file_path.name}")
        else:
            # Fallback to original converter
            conversion_results = self.document_converter.convert_to_markdown(str(file_path))
            if conversion_results:
                if len(conversion_results[0]) == 3:
                    markdown_text, doc_metadata, docling_doc = conversion_results[0]
                else:
                    markdown_text, doc_metadata = conversion_results[0]
            else:
                raise RuntimeError(f"Failed to convert {file_path.name}")

        # Merge metadata
        merged_metadata = {**(document_metadata or {}), **doc_metadata}

        # Chunk document (pass docling_doc for structure-aware chunking)
        chunks = self.chunker.chunk(
            text=markdown_text,
            document_id=document_id,
            document_metadata=merged_metadata,
            docling_doc=docling_doc if hasattr(self.chunker, 'chunk') and 'docling_doc' in self.chunker.chunk.__code__.co_varnames else None
        )

        # Print extraction summary
        if 'tables' in merged_metadata:
            print(f"   Tables extracted: {len(merged_metadata['tables'])}")
        if 'formulas' in merged_metadata:
            print(f"   Formulas extracted: {len(merged_metadata['formulas'])}")
        if 'code_blocks' in merged_metadata:
            print(f"   Code blocks extracted: {len(merged_metadata['code_blocks'])}")

        print(f"✅ Document processed: {len(chunks)} chunks created")

        return chunks, merged_metadata

    def process_batch(
        self,
        file_paths: List[str],
        base_metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.

        Args:
            file_paths: List of file paths
            base_metadata: Base metadata for all documents
            progress_callback: Optional callback(current, total, filename)

        Returns:
            List of processing results
        """
        results = []

        for i, file_path in enumerate(file_paths, 1):
            try:
                file_name = Path(file_path).name
                document_id = f"doc_{i}_{Path(file_path).stem}"

                chunks, metadata = self.process_document(
                    file_path=file_path,
                    document_id=document_id,
                    document_metadata=base_metadata
                )

                results.append({
                    "file_path": file_path,
                    "document_id": document_id,
                    "chunks": chunks,
                    "metadata": metadata,
                    "status": "success"
                })

                if progress_callback:
                    progress_callback(i, len(file_paths), file_name)

            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")
                results.append({
                    "file_path": file_path,
                    "status": "error",
                    "error": str(e)
                })

        return results


def test_integration():
    """Test integrated pipeline."""
    print("\n🧪 Docling Integration Test\n")

    pipeline = DoclingIntegratedPipeline(
        max_chunk_size=1000,
        use_docling_vlm=True,
        use_docling_chunking=True,
        enable_audio=True,
        extract_tables=True,
        extract_formulas=True,
        extract_code=True
    )

    print("\n📋 Supported Formats:")
    for fmt, typ in pipeline.SUPPORTED_FORMATS.items():
        print(f"   {fmt} ({typ})")

    print("\n✅ Pipeline ready for document processing!")


if __name__ == "__main__":
    test_integration()
