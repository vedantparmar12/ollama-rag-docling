"""Max-Min Semantic Chunking for RAG

Implements the Max-Min semantic chunking algorithm that uses semantic
similarity and a Max-Min algorithm to identify semantically coherent text segments.

Based on: "Max–Min semantic chunking of documents for RAG application"
(Discover Computing, 2025) - AMI scores of 0.85-0.90

This method outperforms traditional chunking by:
- Preserving semantic coherence within chunks
- Respecting natural document boundaries
- Achieving 0.56 average accuracy on diverse datasets
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re


class MaxMinSemanticChunker:
    """
    Max-Min semantic chunking using similarity-based boundary detection.

    Algorithm:
    1. Split document into sentences
    2. Embed all sentences
    3. Compute pairwise similarities
    4. Find local minima in similarity curve (chunk boundaries)
    5. Merge small chunks respecting semantic coherence
    """

    def __init__(
        self,
        embedding_model=None,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
        similarity_threshold: float = 0.80,
        boundary_threshold: float = 0.70,
        overlap_sentences: int = 1
    ):
        """
        Initialize Max-Min semantic chunker.

        Args:
            embedding_model: Model for sentence embeddings
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            similarity_threshold: Threshold for grouping similar sentences
            boundary_threshold: Threshold for detecting chunk boundaries
            overlap_sentences: Number of sentences to overlap between chunks
        """
        self.embedding_model = embedding_model
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.boundary_threshold = boundary_threshold
        self.overlap_sentences = overlap_sentences

    def _load_embedding_model(self):
        """Lazy load embedding model."""
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            print("🔧 Loading embedding model for semantic chunking...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded")
        return self.embedding_model

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using multiple delimiters.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Try to use NLTK first
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("📥 Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)

            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback to regex-based splitting
            sentence_endings = r'[.!?]+[\s"]'
            sentences = re.split(sentence_endings, text)
            sentences = [s.strip() for s in sentences if s.strip()]

        # Filter out very short sentences (< 10 chars)
        sentences = [s for s in sentences if len(s) >= 10]

        return sentences

    def _compute_similarity_curve(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity curve between consecutive sentences.

        Args:
            embeddings: Sentence embeddings (N x D)

        Returns:
            Similarity scores between consecutive sentences (N-1,)
        """
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)

        return np.array(similarities)

    def _find_boundaries(
        self,
        similarities: np.ndarray,
        sentences: List[str]
    ) -> List[int]:
        """
        Find chunk boundaries using Max-Min algorithm.

        A boundary exists where:
        1. Similarity drops below threshold (local minimum)
        2. Chunk size constraints are satisfied

        Args:
            similarities: Similarity curve
            sentences: List of sentences

        Returns:
            List of boundary indices
        """
        boundaries = [0]  # Always start with first sentence

        i = 0
        current_chunk_size = 0

        while i < len(similarities):
            current_chunk_size += len(sentences[i])

            # Check if we should create a boundary
            is_low_similarity = similarities[i] < self.boundary_threshold
            is_local_minimum = False

            # Check if this is a local minimum
            if i > 0 and i < len(similarities) - 1:
                is_local_minimum = (
                    similarities[i] < similarities[i - 1] and
                    similarities[i] < similarities[i + 1]
                )

            # Check size constraints
            exceeds_min_size = current_chunk_size >= self.min_chunk_size
            exceeds_max_size = current_chunk_size >= self.max_chunk_size

            # Create boundary if conditions met
            if exceeds_max_size or (exceeds_min_size and (is_low_similarity or is_local_minimum)):
                boundaries.append(i + 1)
                current_chunk_size = 0

            i += 1

        # Always include the last sentence
        if boundaries[-1] != len(sentences):
            boundaries.append(len(sentences))

        return boundaries

    def _merge_small_chunks(
        self,
        chunks: List[str],
        embeddings_by_chunk: List[np.ndarray]
    ) -> Tuple[List[str], List[np.ndarray]]:
        """
        Merge chunks that are too small with neighboring chunks.

        Args:
            chunks: List of text chunks
            embeddings_by_chunk: Embeddings for each chunk

        Returns:
            Merged chunks and their embeddings
        """
        merged_chunks = []
        merged_embeddings = []

        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            current_embedding = embeddings_by_chunk[i]

            # If chunk is too small, try to merge with next
            if len(current_chunk) < self.min_chunk_size and i < len(chunks) - 1:
                # Compute similarity with next chunk
                next_embedding = embeddings_by_chunk[i + 1]
                sim = cosine_similarity([current_embedding], [next_embedding])[0][0]

                # Merge if similar enough
                if sim >= self.similarity_threshold:
                    merged_chunk = current_chunk + " " + chunks[i + 1]
                    merged_embedding = (current_embedding + next_embedding) / 2
                    merged_chunks.append(merged_chunk)
                    merged_embeddings.append(merged_embedding)
                    i += 2  # Skip next chunk
                    continue

            merged_chunks.append(current_chunk)
            merged_embeddings.append(current_embedding)
            i += 1

        return merged_chunks, merged_embeddings

    def chunk_text(self, text: str, document_id: str = "doc") -> List[Dict[str, Any]]:
        """
        Chunk text using Max-Min semantic chunking.

        Args:
            text: Input text to chunk
            document_id: ID for the document

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)

        if len(sentences) == 0:
            return []

        if len(sentences) == 1:
            return [{
                'text': sentences[0],
                'chunk_id': f"{document_id}_chunk_0",
                'metadata': {
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'char_count': len(sentences[0]),
                    'chunking_method': 'maxmin_semantic'
                }
            }]

        # Load embedding model and embed sentences
        model = self._load_embedding_model()
        print(f"📊 Embedding {len(sentences)} sentences for semantic analysis...")
        embeddings = model.encode(sentences, show_progress_bar=False)

        # Compute similarity curve
        similarities = self._compute_similarity_curve(embeddings)

        # Find boundaries
        boundaries = self._find_boundaries(similarities, sentences)

        # Create initial chunks
        initial_chunks = []
        chunk_embeddings = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            # Compute chunk embedding (average of sentence embeddings)
            chunk_embedding = np.mean(embeddings[start_idx:end_idx], axis=0)

            initial_chunks.append(chunk_text)
            chunk_embeddings.append(chunk_embedding)

        # Merge small chunks
        final_chunks, final_embeddings = self._merge_small_chunks(
            initial_chunks,
            chunk_embeddings
        )

        # Create output with metadata
        output_chunks = []
        for i, (chunk_text, chunk_emb) in enumerate(zip(final_chunks, final_embeddings)):
            # Add overlap with previous chunk if requested
            if i > 0 and self.overlap_sentences > 0:
                prev_sentences = sentences[max(0, boundaries[i] - self.overlap_sentences):boundaries[i]]
                overlap_text = " ".join(prev_sentences)
                chunk_text = overlap_text + " " + chunk_text

            output_chunks.append({
                'text': chunk_text,
                'chunk_id': f"{document_id}_chunk_{i}",
                'embedding': chunk_emb,
                'metadata': {
                    'chunk_index': i,
                    'total_chunks': len(final_chunks),
                    'char_count': len(chunk_text),
                    'sentence_count': chunk_text.count('.') + chunk_text.count('!') + chunk_text.count('?'),
                    'chunking_method': 'maxmin_semantic',
                    'avg_similarity': float(np.mean(similarities)) if len(similarities) > 0 else 1.0
                }
            })

        print(f"✅ Created {len(output_chunks)} semantic chunks (avg: {sum(len(c['text']) for c in output_chunks) // len(output_chunks)} chars)")

        return output_chunks


# Example usage and testing
if __name__ == "__main__":
    # Test text with multiple topics
    test_text = """
    The capital of France is Paris. Paris is known for the Eiffel Tower.
    The city has a rich history dating back to the Middle Ages.

    Machine learning is a subset of artificial intelligence. It focuses on
    algorithms that learn from data. Deep learning is a type of machine learning
    that uses neural networks.

    Climate change is a global challenge. Rising temperatures are affecting
    ecosystems worldwide. Governments are working on solutions to reduce
    carbon emissions.

    Python is a popular programming language. It is used for web development,
    data science, and automation. Many companies use Python in production.
    """

    chunker = MaxMinSemanticChunker(
        min_chunk_size=100,
        max_chunk_size=500,
        similarity_threshold=0.80,
        boundary_threshold=0.70
    )

    chunks = chunker.chunk_text(test_text, document_id="test_doc")

    print("\n" + "="*60)
    print("MAX-MIN SEMANTIC CHUNKING RESULTS")
    print("="*60)

    for chunk in chunks:
        print(f"\nChunk {chunk['metadata']['chunk_index'] + 1}/{chunk['metadata']['total_chunks']}:")
        print(f"  Chars: {chunk['metadata']['char_count']}")
        print(f"  Sentences: ~{chunk['metadata']['sentence_count']}")
        print(f"  Text: {chunk['text'][:100]}...")

    print(f"\n✅ Semantic chunking test completed")
    print(f"   Average chunk size: {sum(c['metadata']['char_count'] for c in chunks) // len(chunks)} chars")
    print(f"   Size variance: {np.std([c['metadata']['char_count'] for c in chunks]):.1f}")
