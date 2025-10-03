"""Multimodal Embedding System for Specialized Content Types

Provides specialized embeddings for different content types:
- Text: General text embedding (Qwen3-Embedding)
- Code: Code-specific embedding (CodeBERT)
- Tables: Table-specific embedding (ColBERT)
- Formulas: Math formula embedding

This improves retrieval accuracy by using domain-specific models.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from enum import Enum


class ContentType(Enum):
    """Content type enumeration."""
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    FORMULA = "formula"
    MIXED = "mixed"


class MultiModalEmbedder:
    """
    Multi-modal embedder that routes content to specialized models.

    Uses different embedding models based on content type:
    - Text → Qwen3-Embedding-0.6B
    - Code → CodeBERT
    - Tables → ColBERT (token-level)
    - Formulas → Same as text (can be extended)
    """

    def __init__(
        self,
        text_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        code_model_name: str = "microsoft/codebert-base",
        enable_code: bool = True,
        enable_table: bool = True,
        device: str = None
    ):
        """
        Initialize multi-modal embedder.

        Args:
            text_model_name: Model for general text
            code_model_name: Model for code chunks
            enable_code: Whether to use specialized code embeddings
            enable_table: Whether to use specialized table embeddings
            device: Device to use (cuda/cpu/mps)
        """
        self.text_model_name = text_model_name
        self.code_model_name = code_model_name
        self.enable_code = enable_code
        self.enable_table = enable_table

        # Auto-detect device
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Lazy-load models
        self._text_model = None
        self._code_model = None
        self._table_embedder = None

        print(f"✅ MultiModalEmbedder initialized (device: {self.device})")

    def _load_text_model(self):
        """Lazy load text embedding model."""
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"🔧 Loading text embedding model: {self.text_model_name}")
            self._text_model = SentenceTransformer(self.text_model_name, device=self.device)
            print("✅ Text model loaded")
        return self._text_model

    def _load_code_model(self):
        """Lazy load code embedding model."""
        if not self.enable_code:
            return self._load_text_model()  # Fallback to text model

        if self._code_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"🔧 Loading code embedding model: {self.code_model_name}")
                self._code_model = SentenceTransformer(self.code_model_name, device=self.device)
                print("✅ Code model loaded")
            except Exception as e:
                print(f"⚠️ Failed to load code model: {e}")
                print("   Falling back to text model for code")
                self._code_model = self._load_text_model()

        return self._code_model

    def _load_table_embedder(self):
        """Lazy load table-specific embedder (ColBERT)."""
        if not self.enable_table:
            return None

        if self._table_embedder is None:
            try:
                from rerankers import Reranker
                print("🔧 Loading table embedder (ColBERT)...")
                self._table_embedder = Reranker(
                    "answerdotai/answerai-colbert-small-v1",
                    model_type='colbert',
                    device=self.device
                )
                print("✅ Table embedder loaded")
            except Exception as e:
                print(f"⚠️ Failed to load table embedder: {e}")
                print("   Tables will use text embeddings")
                self._table_embedder = None

        return self._table_embedder

    def detect_content_type(self, text: str, metadata: Optional[Dict] = None) -> ContentType:
        """
        Detect content type from text and metadata.

        Args:
            text: Content text
            metadata: Optional metadata with type hints

        Returns:
            Detected ContentType
        """
        # Check metadata first
        if metadata:
            if metadata.get('type') == 'code' or metadata.get('is_code'):
                return ContentType.CODE
            if metadata.get('type') == 'table' or metadata.get('is_table'):
                return ContentType.TABLE
            if metadata.get('type') == 'formula' or metadata.get('is_formula'):
                return ContentType.FORMULA

        # Heuristic detection from text
        text_lower = text.lower()

        # Code detection
        code_indicators = [
            'def ', 'class ', 'import ', 'function ',
            '=>', 'const ', 'let ', 'var ',
            'public ', 'private ', 'protected ',
            '{', '}', ');', 'return ', 'async '
        ]
        code_score = sum(1 for indicator in code_indicators if indicator in text_lower)

        if code_score >= 3:
            return ContentType.CODE

        # Table detection (Markdown tables)
        if '|' in text and text.count('|') >= 4:
            lines = text.split('\n')
            table_lines = [line for line in lines if '|' in line]
            if len(table_lines) >= 2:
                return ContentType.TABLE

        # Formula detection
        if '$$' in text or ('$' in text and any(c in text for c in ['\\sum', '\\int', '\\frac', '='])):
            return ContentType.FORMULA

        return ContentType.TEXT

    def embed_single(
        self,
        text: str,
        content_type: Optional[ContentType] = None,
        metadata: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Embed single text using appropriate model.

        Args:
            text: Text to embed
            content_type: Explicit content type (auto-detect if None)
            metadata: Optional metadata

        Returns:
            Embedding vector
        """
        # Auto-detect if not specified
        if content_type is None:
            content_type = self.detect_content_type(text, metadata)

        # Route to appropriate model
        if content_type == ContentType.CODE:
            model = self._load_code_model()
            embedding = model.encode(text, show_progress_bar=False)
        elif content_type == ContentType.TABLE:
            # Tables use text model (ColBERT is for reranking)
            model = self._load_text_model()
            embedding = model.encode(text, show_progress_bar=False)
        else:  # TEXT, FORMULA, MIXED
            model = self._load_text_model()
            embedding = model.encode(text, show_progress_bar=False)

        return embedding

    def embed_batch(
        self,
        texts: List[str],
        content_types: Optional[List[ContentType]] = None,
        metadata_list: Optional[List[Dict]] = None,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Embed batch of texts using appropriate models.

        Args:
            texts: List of texts
            content_types: List of content types (auto-detect if None)
            metadata_list: List of metadata dicts
            batch_size: Batch size for embedding

        Returns:
            Array of embeddings
        """
        # Auto-detect content types if not provided
        if content_types is None:
            content_types = []
            for i, text in enumerate(texts):
                meta = metadata_list[i] if metadata_list else None
                content_types.append(self.detect_content_type(text, meta))

        # Group by content type for efficient batching
        type_groups: Dict[ContentType, List[int]] = {}
        for idx, ctype in enumerate(content_types):
            if ctype not in type_groups:
                type_groups[ctype] = []
            type_groups[ctype].append(idx)

        # Embed each group with appropriate model
        embeddings = np.zeros((len(texts), 1024))  # Assuming 1024-dim embeddings

        for ctype, indices in type_groups.items():
            group_texts = [texts[i] for i in indices]

            if ctype == ContentType.CODE and self.enable_code:
                model = self._load_code_model()
            else:
                model = self._load_text_model()

            group_embeddings = model.encode(
                group_texts,
                batch_size=batch_size,
                show_progress_bar=len(group_texts) > 50
            )

            # Place embeddings back in original order
            for i, idx in enumerate(indices):
                embeddings[idx] = group_embeddings[i]

        return embeddings

    def embed_with_metadata(
        self,
        chunks: List[Dict[str, Any]],
        text_field: str = 'text',
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Embed chunks and add embeddings to metadata.

        Args:
            chunks: List of chunk dicts with text and metadata
            text_field: Field name containing text
            batch_size: Batch size for embedding

        Returns:
            Chunks with added 'embedding' field
        """
        texts = [chunk[text_field] for chunk in chunks]
        metadata_list = [chunk.get('metadata', {}) for chunk in chunks]

        # Detect content types
        content_types = [
            self.detect_content_type(text, meta)
            for text, meta in zip(texts, metadata_list)
        ]

        # Embed all texts
        embeddings = self.embed_batch(texts, content_types, metadata_list, batch_size)

        # Add embeddings and content types to chunks
        enriched_chunks = []
        for chunk, embedding, ctype in zip(chunks, embeddings, content_types):
            enriched_chunk = chunk.copy()
            enriched_chunk['embedding'] = embedding
            enriched_chunk['content_type'] = ctype.value
            enriched_chunks.append(enriched_chunk)

        return enriched_chunks


# Example usage
if __name__ == "__main__":
    # Test multi-modal embedding
    embedder = MultiModalEmbedder(enable_code=True, enable_table=True)

    # Test samples
    samples = [
        {
            "text": "The capital of France is Paris.",
            "metadata": {}
        },
        {
            "text": "def hello_world():\n    print('Hello, World!')\n    return True",
            "metadata": {"type": "code"}
        },
        {
            "text": "| Name | Age | City |\n|------|-----|------|\n| John | 30 | NYC |",
            "metadata": {}
        }
    ]

    # Embed with automatic type detection
    enriched = embedder.embed_with_metadata(samples)

    print("\n" + "="*60)
    print("MULTIMODAL EMBEDDING RESULTS")
    print("="*60)

    for i, chunk in enumerate(enriched):
        print(f"\nChunk {i+1}:")
        print(f"  Content Type: {chunk['content_type']}")
        print(f"  Embedding Shape: {chunk['embedding'].shape}")
        print(f"  Text Preview: {chunk['text'][:50]}...")

    print("\n✅ Multi-modal embedding test completed")
