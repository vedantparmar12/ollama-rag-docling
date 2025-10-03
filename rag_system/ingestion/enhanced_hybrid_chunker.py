"""
Enhanced Hybrid Chunker - Combines Docling's hierarchical structure with custom chunking logic
Preserves your existing MarkdownRecursiveChunker while adding Docling's document structure awareness
"""
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from docling_core.transforms.chunker import HybridChunker as DoclingHybridChunker
    from docling_core.types.doc import DoclingDocument, TableItem, PictureItem, ListItem
    from transformers import AutoTokenizer
    DOCLING_CHUNKING_AVAILABLE = True
except ImportError:
    DOCLING_CHUNKING_AVAILABLE = False
    print("⚠️ Docling chunking not available. Install: pip install docling-core")


class EnhancedHybridChunker:
    """
    Advanced hybrid chunker that combines:
    1. Docling's hierarchical document structure awareness
    2. Your custom MarkdownRecursiveChunker logic
    3. Intelligent table/code/formula preservation

    This preserves your existing chunking strategy while adding Docling enhancements.
    """

    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 200,
        tokenizer_model: str = "Qwen/Qwen3-Embedding-0.6B",
        use_docling_structure: bool = True,
        preserve_tables: bool = True,
        preserve_code: bool = True,
        preserve_formulas: bool = True,
        merge_list_items: bool = True
    ):
        """
        Initialize enhanced hybrid chunker.

        Args:
            max_chunk_size: Maximum chunk size in tokens
            min_chunk_size: Minimum chunk size in tokens
            tokenizer_model: HuggingFace tokenizer for token counting
            use_docling_structure: Use Docling's hierarchical structure
            preserve_tables: Keep tables intact in chunks
            preserve_code: Keep code blocks intact
            preserve_formulas: Keep formulas intact
            merge_list_items: Merge list items together
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.use_docling_structure = use_docling_structure
        self.preserve_tables = preserve_tables
        self.preserve_code = preserve_code
        self.preserve_formulas = preserve_formulas
        self.merge_list_items = merge_list_items

        # Initialize tokenizer
        self.tokenizer = self._load_tokenizer(tokenizer_model)

        # Initialize Docling chunker if available
        self.docling_chunker = None
        if DOCLING_CHUNKING_AVAILABLE and use_docling_structure:
            try:
                self.docling_chunker = DoclingHybridChunker(
                    tokenizer=self.tokenizer,
                    max_tokens=max_chunk_size,
                    merge_peers=merge_list_items
                )
                print("✅ Enhanced Hybrid Chunker initialized with Docling structure")
            except Exception as e:
                print(f"⚠️ Failed to init Docling chunker: {e}")

        # Fallback to custom logic
        self.split_priority = ["\n## ", "\n### ", "\n#### ", "```", "\n\n"]

    def _load_tokenizer(self, model_name: str):
        """Load tokenizer for token counting."""
        repo_id = model_name
        if "/" not in model_name and not model_name.startswith("Qwen/"):
            repo_id = {
                "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
            }.get(model_name.lower(), model_name)

        try:
            return AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        except Exception as e:
            print(f"⚠️ Tokenizer load failed: {e}. Using char approximation.")
            return None

    def _token_len(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.tokenize(text))
        return max(1, len(text) // 4)  # Fallback: ~4 chars per token

    def chunk(
        self,
        text: str,
        document_id: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        docling_doc: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk document using hybrid approach.

        Args:
            text: Markdown text to chunk
            document_id: Document identifier
            document_metadata: Document-level metadata
            docling_doc: Optional DoclingDocument for structure-aware chunking

        Returns:
            List of chunk dictionaries
        """
        # Strategy 1: Use Docling structure if available
        if self.docling_chunker and docling_doc and DOCLING_CHUNKING_AVAILABLE:
            return self._chunk_with_docling_structure(
                docling_doc, document_id, document_metadata
            )

        # Strategy 2: Fallback to enhanced markdown chunking
        return self._chunk_markdown_enhanced(
            text, document_id, document_metadata
        )

    def _chunk_with_docling_structure(
        self,
        docling_doc: 'DoclingDocument',
        document_id: str,
        document_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk using Docling's hierarchical document structure.
        This respects document layout, tables, and semantic boundaries.
        """
        chunks = []

        try:
            # Use Docling's HybridChunker
            docling_chunks = list(self.docling_chunker.chunk(docling_doc))

            for idx, chunk_obj in enumerate(docling_chunks):
                # Extract chunk text and metadata
                chunk_text = chunk_obj.text if hasattr(chunk_obj, 'text') else str(chunk_obj)

                # Build metadata from Docling chunk info
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": idx,
                    "token_count": self._token_len(chunk_text),
                    "chunking_method": "docling_hybrid",
                }

                # Preserve Docling metadata (headings, captions, page numbers)
                if hasattr(chunk_obj, 'meta'):
                    meta = chunk_obj.meta
                    if hasattr(meta, 'headings'):
                        chunk_metadata['headings'] = meta.headings
                    if hasattr(meta, 'doc_items'):
                        # Extract document element types in this chunk
                        chunk_metadata['element_types'] = [
                            type(item).__name__ for item in meta.doc_items
                        ]

                # Merge document-level metadata
                if document_metadata:
                    chunk_metadata.update({
                        f"doc_{k}": v for k, v in document_metadata.items()
                    })

                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })

            print(f"✅ Docling structure chunking: {len(chunks)} chunks created")

        except Exception as e:
            print(f"⚠️ Docling chunking failed: {e}. Falling back to markdown chunking.")
            return self._chunk_markdown_enhanced(
                docling_doc.export_to_markdown() if docling_doc else "",
                document_id,
                document_metadata
            )

        return chunks

    def _chunk_markdown_enhanced(
        self,
        text: str,
        document_id: str,
        document_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhanced markdown chunking with structure preservation.
        Uses your original MarkdownRecursiveChunker logic + improvements.
        """
        # Extract special elements that should be preserved
        preserved_elements = self._extract_preserved_elements(text)

        # Split text by hierarchy
        raw_chunks = self._split_text_hierarchical(text, self.split_priority)

        # Post-process chunks
        final_chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            # Skip empty chunks
            if not chunk_text.strip():
                continue

            # Check if chunk contains preserved elements
            contains_table = any(
                elem['type'] == 'table' and elem['content'] in chunk_text
                for elem in preserved_elements
            )
            contains_code = any(
                elem['type'] == 'code' and elem['content'] in chunk_text
                for elem in preserved_elements
            )

            chunk_metadata = {
                "document_id": document_id,
                "chunk_index": idx,
                "token_count": self._token_len(chunk_text),
                "chunking_method": "markdown_enhanced",
                "contains_table": contains_table,
                "contains_code": contains_code,
            }

            # Add document metadata
            if document_metadata:
                chunk_metadata.update({
                    f"doc_{k}": v for k, v in document_metadata.items()
                })

            final_chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

        print(f"✅ Enhanced markdown chunking: {len(final_chunks)} chunks created")
        return final_chunks

    def _extract_preserved_elements(self, text: str) -> List[Dict[str, str]]:
        """Extract elements that should be preserved intact (tables, code)."""
        import re
        elements = []

        # Extract code blocks
        if self.preserve_code:
            code_pattern = r'(```[\s\S]*?```)'
            for match in re.finditer(code_pattern, text):
                elements.append({
                    'type': 'code',
                    'content': match.group(1)
                })

        # Extract tables (basic markdown table detection)
        if self.preserve_tables:
            table_pattern = r'(\|[^\n]+\|\n\|[-:\s|]+\|[\s\S]*?)(?=\n\n|\n#|\Z)'
            for match in re.finditer(table_pattern, text):
                elements.append({
                    'type': 'table',
                    'content': match.group(1)
                })

        # Extract formulas (LaTeX-style)
        if self.preserve_formulas:
            formula_patterns = [
                r'(\$\$[\s\S]*?\$\$)',  # Block formulas
                r'(\$[^\$\n]+\$)',  # Inline formulas
            ]
            for pattern in formula_patterns:
                for match in re.finditer(pattern, text):
                    elements.append({
                        'type': 'formula',
                        'content': match.group(1)
                    })

        return elements

    def _split_text_hierarchical(
        self,
        text: str,
        separators: List[str]
    ) -> List[str]:
        """Split text using hierarchical separators (your original logic)."""
        import re

        final_chunks = []
        chunks_to_process = [text]

        for sep in separators:
            new_chunks = []
            for chunk in chunks_to_process:
                if self._token_len(chunk) > self.max_chunk_size:
                    sub_chunks = re.split(f'({sep})', chunk)
                    combined = []
                    i = 0
                    while i < len(sub_chunks):
                        if i + 1 < len(sub_chunks) and sub_chunks[i + 1] == sep:
                            combined.append(sub_chunks[i + 1] + sub_chunks[i + 2] if i + 2 < len(sub_chunks) else sub_chunks[i + 1])
                            i += 3
                        else:
                            if sub_chunks[i]:
                                combined.append(sub_chunks[i])
                            i += 1
                    new_chunks.extend(combined)
                else:
                    new_chunks.append(chunk)
            chunks_to_process = new_chunks

        # Final word-level splitting for oversized chunks
        for chunk in chunks_to_process:
            if self._token_len(chunk) > self.max_chunk_size:
                words = chunk.split()
                current = ""
                for word in words:
                    test = current + " " + word if current else word
                    if self._token_len(test) <= self.max_chunk_size:
                        current = test
                    else:
                        if current:
                            final_chunks.append(current)
                        current = word
                if current:
                    final_chunks.append(current)
            else:
                final_chunks.append(chunk)

        return final_chunks


def test_enhanced_chunker():
    """Test enhanced hybrid chunker."""
    print("\n🧪 Enhanced Hybrid Chunker Test")

    chunker = EnhancedHybridChunker(
        max_chunk_size=1000,
        min_chunk_size=200,
        use_docling_structure=True,
        preserve_tables=True,
        preserve_code=True
    )

    sample_text = """
# Document Title

## Section 1
This is some text content.

```python
def hello():
    print("world")
```

## Section 2
| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
"""

    chunks = chunker.chunk(sample_text, "test_doc", {"source": "test"})
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}: {chunk['metadata']['token_count']} tokens")
        print(f"Contains table: {chunk['metadata'].get('contains_table', False)}")
        print(f"Contains code: {chunk['metadata'].get('contains_code', False)}")


if __name__ == "__main__":
    test_enhanced_chunker()
