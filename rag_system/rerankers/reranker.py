from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Dict, Any

class QwenReranker:
    """
    A reranker that uses a local Hugging Face transformer model.

    Supports multiple reranker types:
    - Cross-encoder models (BAAI/bge-reranker-base)
    - ColBERT models (jinaai/jina-colbert-v2, answerdotai/answerai-colbert-small-v1)

    Automatically detects model type and uses appropriate strategy.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", use_colbert: bool = None):
        self.model_name = model_name

        # Auto-detect ColBERT models if not specified
        if use_colbert is None:
            self.use_colbert = 'colbert' in model_name.lower() or 'answerai' in model_name.lower()
        else:
            self.use_colbert = use_colbert

        # Auto-select the best available device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Initialize appropriate model type
        if self.use_colbert:
            self._init_colbert_model()
        else:
            self._init_cross_encoder_model()

    def _init_cross_encoder_model(self):
        """Initialize traditional cross-encoder reranker."""
        print(f"Initializing Cross-Encoder Reranker with model '{self.model_name}' on device '{self.device}'.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else None,
        ).to(self.device).eval()
        self.max_length = 512  # Standard cross-encoder limit
        print("✅ Cross-Encoder Reranker loaded successfully.")

    def _init_colbert_model(self):
        """Initialize ColBERT reranker using rerankers library."""
        try:
            from rerankers import Reranker
            print(f"Initializing ColBERT Reranker with model '{self.model_name}'...")
            self.colbert_reranker = Reranker(self.model_name, model_type='colbert', device=self.device)

            # Determine max length based on model
            if 'jina-colbert-v2' in self.model_name:
                self.max_length = 8192  # Jina-ColBERT-v2 supports long context
            else:
                self.max_length = 512  # answerai-colbert-small and others

            print(f"✅ ColBERT Reranker loaded successfully (max_length: {self.max_length}).")
        except ImportError:
            print("⚠️ 'rerankers' library not found. Falling back to cross-encoder mode.")
            print("   Install with: pip install rerankers")
            self.use_colbert = False
            self._init_cross_encoder_model()
        except Exception as e:
            print(f"⚠️ Failed to load ColBERT model: {e}")
            print("   Falling back to cross-encoder mode.")
            self.use_colbert = False
            self._init_cross_encoder_model()

    def _format_instruction(self, query: str, doc: str):
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5, *, early_exit: bool = True, margin: float = 0.4, min_scored: int = 8, batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on their relevance to a query.

        For ColBERT models: Uses late-interaction mechanism for efficient token-level matching.
        For Cross-Encoders: Uses early-exit optimization for speed.

        Args:
            query: Search query string
            documents: List of document dicts with 'text' field
            top_k: Number of top results to return
            early_exit: Enable early stopping (cross-encoder only)
            margin: Score margin for early exit
            min_scored: Minimum docs to score before early exit
            batch_size: Batch size for processing

        Returns:
            Reranked documents with 'rerank_score' field
        """
        if not documents:
            return []

        # Route to appropriate reranker
        if self.use_colbert:
            return self._rerank_colbert(query, documents, top_k)
        else:
            return self._rerank_cross_encoder(query, documents, top_k, early_exit, margin, min_scored, batch_size)

    def _rerank_colbert(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank using ColBERT late-interaction."""
        if not hasattr(self, 'colbert_reranker'):
            print("⚠️ ColBERT reranker not initialized, returning original order")
            return documents[:top_k]

        # Extract texts and truncate to max_length
        texts = [doc['text'][:self.max_length] for doc in documents]

        try:
            # ColBERT reranking
            results = self.colbert_reranker.rank(query, texts)

            # Map results back to original documents
            reranked_docs: List[Dict[str, Any]] = []
            for result in results[:top_k]:
                doc = documents[result.doc_id].copy()
                doc['rerank_score'] = result.score
                reranked_docs.append(doc)

            return reranked_docs

        except Exception as e:
            print(f"⚠️ ColBERT reranking failed: {e}, returning original order")
            return documents[:top_k]

    def _rerank_cross_encoder(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int,
        early_exit: bool,
        margin: float,
        min_scored: int,
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """Rerank using cross-encoder with early-exit optimization."""
        # Sort by the upstream (hybrid) score so that the strongest candidates are evaluated first.
        docs_sorted = sorted(documents, key=lambda d: d.get('score', 0.0), reverse=True)

        scored_pairs: List[tuple[float, Dict[str, Any]]] = []

        with torch.no_grad():
            for start in range(0, len(docs_sorted), batch_size):
                batch_docs = docs_sorted[start : start + batch_size]
                batch_pairs = [[query, d['text']] for d in batch_docs]

                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                ).to(self.device)

                logits = self.model(**inputs).logits.view(-1)
                batch_scores = logits.float().cpu().tolist()

                scored_pairs.extend(zip(batch_scores, batch_docs))

                # --- Early-exit check ---
                if early_exit and len(scored_pairs) >= min_scored:
                    # Current best and worst among *already* scored docs
                    best_score = max(scored_pairs, key=lambda x: x[0])[0]
                    worst_score = min(scored_pairs, key=lambda x: x[0])[0]
                    if best_score - worst_score >= margin:
                        break

        # Sort final set and attach scores
        sorted_by_score = sorted(scored_pairs, key=lambda x: x[0], reverse=True)
        reranked_docs: List[Dict[str, Any]] = []
        for score, doc in sorted_by_score[:top_k]:
            doc_with_score = doc.copy()
            doc_with_score['rerank_score'] = score
            reranked_docs.append(doc_with_score)

        return reranked_docs

if __name__ == '__main__':
    # This test requires an internet connection to download the models.
    try:
        reranker = QwenReranker(model_name="BAAI/bge-reranker-base")
        
        query = "What is the capital of France?"
        documents = [
            {'text': "Paris is the capital of France.", 'metadata': {'doc_id': 'a'}},
            {'text': "The Eiffel Tower is in Paris.", 'metadata': {'doc_id': 'b'}},
            {'text': "France is a country in Europe.", 'metadata': {'doc_id': 'c'}},
        ]
        
        reranked_documents = reranker.rerank(query, documents)
        
        print("\n--- Verification ---")
        print(f"Query: {query}")
        print("Reranked documents:")
        for doc in reranked_documents:
            print(f"  - Score: {doc['rerank_score']:.4f}, Text: {doc['text']}")

    except Exception as e:
        print(f"\nAn error occurred during the QwenReranker test: {e}")
        print("Please ensure you have an internet connection for model downloads.")
