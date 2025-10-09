"""
Comprehensive RAG Evaluation Framework
Combines multiple evaluation methods for robust assessment

Includes:
1. LLM-Free Metrics (fast, free, deterministic)
2. Embedding-Based Metrics (semantic similarity)
3. LLM-as-Judge Metrics (using local Ollama)
4. Custom Domain Metrics
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util
import json
from dataclasses import dataclass, asdict
import time


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    query: str
    response: str
    contexts: List[str]
    scores: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: float

    def to_dict(self) -> Dict:
        return asdict(self)


class ComprehensiveRAGEvaluator:
    """
    All-in-one RAG evaluator with multiple metric types.

    Metrics Categories:
    1. LLM-Free (0 cost, instant)
    2. Embedding-Based (low cost, fast)
    3. LLM-Judge (uses local Ollama, free)
    4. Custom Domain Metrics
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "llama3.2"
    ):
        """
        Initialize evaluator.

        Args:
            embedding_model: Model for semantic similarity
            ollama_host: Ollama API endpoint
            ollama_model: Local LLM for judge metrics
        """
        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model)

        # Ollama config
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model

    # ========================================================================
    # 1. LLM-FREE METRICS (Instant, Zero Cost)
    # ========================================================================

    def exact_match(self, response: str, expected: str) -> float:
        """Binary exact match"""
        return 1.0 if response.strip().lower() == expected.strip().lower() else 0.0

    def token_overlap(self, response: str, expected: str) -> float:
        """Token-level overlap (F1)"""
        response_tokens = set(response.lower().split())
        expected_tokens = set(expected.lower().split())

        if not expected_tokens:
            return 0.0

        intersection = response_tokens & expected_tokens
        precision = len(intersection) / len(response_tokens) if response_tokens else 0
        recall = len(intersection) / len(expected_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def keyword_coverage(self, response: str, required_keywords: List[str]) -> float:
        """Check if response contains required keywords"""
        response_lower = response.lower()
        found = sum(1 for kw in required_keywords if kw.lower() in response_lower)
        return found / len(required_keywords) if required_keywords else 0.0

    def response_length_score(self, response: str, ideal_length: int = 100) -> float:
        """Score based on response length (penalize too short/long)"""
        actual_length = len(response.split())

        if actual_length == 0:
            return 0.0

        # Ideal range: 50-150% of ideal_length
        min_len = ideal_length * 0.5
        max_len = ideal_length * 1.5

        if min_len <= actual_length <= max_len:
            return 1.0
        elif actual_length < min_len:
            return actual_length / min_len
        else:
            return max_len / actual_length

    def context_usage_score(self, response: str, contexts: List[str]) -> float:
        """Measure how much context was used in response"""
        if not contexts:
            return 0.0

        response_lower = response.lower()
        context_text = " ".join(contexts).lower()

        # Count overlapping n-grams (3-grams)
        from nltk.util import ngrams
        response_ngrams = set(ngrams(response_lower.split(), 3))
        context_ngrams = set(ngrams(context_text.split(), 3))

        if not response_ngrams:
            return 0.0

        overlap = response_ngrams & context_ngrams
        return len(overlap) / len(response_ngrams)

    # ========================================================================
    # 2. EMBEDDING-BASED METRICS (Fast, Low Cost)
    # ========================================================================

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between embeddings"""
        emb1 = self.embedder.encode(text1, convert_to_tensor=True)
        emb2 = self.embedder.encode(text2, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()

    def answer_relevance(self, query: str, response: str) -> float:
        """How relevant is the answer to the question?"""
        return self.semantic_similarity(query, response)

    def context_relevance(self, query: str, contexts: List[str]) -> float:
        """Average relevance of retrieved contexts to query"""
        if not contexts:
            return 0.0

        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        context_embs = self.embedder.encode(contexts, convert_to_tensor=True)

        similarities = util.cos_sim(query_emb, context_embs)[0]
        return float(similarities.mean())

    def context_precision(self, contexts: List[str], response: str) -> float:
        """How much of retrieved context is actually used?"""
        if not contexts:
            return 0.0

        response_emb = self.embedder.encode(response, convert_to_tensor=True)
        context_embs = self.embedder.encode(contexts, convert_to_tensor=True)

        similarities = util.cos_sim(response_emb, context_embs)[0]

        # Contexts with similarity > 0.5 are considered "used"
        used_contexts = (similarities > 0.5).sum().item()
        return used_contexts / len(contexts)

    def groundedness(self, response: str, contexts: List[str]) -> float:
        """Is the response grounded in the provided contexts?"""
        if not contexts:
            return 0.0

        context_text = " ".join(contexts)
        return self.semantic_similarity(response, context_text)

    # ========================================================================
    # 3. LLM-AS-JUDGE METRICS (Uses Local Ollama - Free!)
    # ========================================================================

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama LLM"""
        import requests

        response = requests.post(
            f"{self.ollama_host}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()['response']

    def hallucination_score(self, response: str, contexts: List[str]) -> float:
        """
        Detect hallucinations using LLM judge.
        Returns: 0.0 (hallucinated) to 1.0 (grounded)
        """
        context_text = "\n".join(contexts)

        prompt = f"""You are evaluating a RAG system response for hallucinations.

Context:
{context_text}

Response:
{response}

Question: Is the response fully grounded in the context, or does it contain hallucinated information?

Answer with ONLY a score from 0 to 10:
- 0 = Completely hallucinated (no basis in context)
- 5 = Partially grounded with some hallucination
- 10 = Fully grounded in context

Score (0-10):"""

        try:
            result = self._call_ollama(prompt)
            # Extract number from response
            import re
            score = re.search(r'\b([0-9]|10)\b', result)
            return float(score.group(1)) / 10.0 if score else 0.5
        except Exception as e:
            print(f"LLM judge error: {e}")
            return 0.5  # Default to uncertain

    def completeness_score(self, query: str, response: str, expected: str = None) -> float:
        """
        Does the response fully answer the question?
        Returns: 0.0 (incomplete) to 1.0 (complete)
        """
        prompt = f"""Evaluate if the response fully answers the question.

Question: {query}
Response: {response}
{f"Expected Answer: {expected}" if expected else ""}

Does the response fully answer the question?

Answer with ONLY a score from 0 to 10:
- 0 = Doesn't answer at all
- 5 = Partially answers
- 10 = Fully complete answer

Score (0-10):"""

        try:
            result = self._call_ollama(prompt)
            import re
            score = re.search(r'\b([0-9]|10)\b', result)
            return float(score.group(1)) / 10.0 if score else 0.5
        except Exception as e:
            print(f"LLM judge error: {e}")
            return 0.5

    # ========================================================================
    # 4. COMPREHENSIVE EVALUATION
    # ========================================================================

    def evaluate_all(
        self,
        query: str,
        response: str,
        contexts: List[str],
        expected_answer: str = None,
        required_keywords: List[str] = None,
        use_llm_judge: bool = True
    ) -> EvaluationResult:
        """
        Run all evaluation metrics.

        Args:
            query: User question
            response: RAG system answer
            contexts: Retrieved context chunks
            expected_answer: Ground truth (optional)
            required_keywords: Keywords that must appear (optional)
            use_llm_judge: Enable LLM-based metrics (slower)

        Returns:
            EvaluationResult with all scores
        """
        scores = {}

        # 1. LLM-Free Metrics (always run - instant)
        if expected_answer:
            scores['exact_match'] = self.exact_match(response, expected_answer)
            scores['token_overlap'] = self.token_overlap(response, expected_answer)

        if required_keywords:
            scores['keyword_coverage'] = self.keyword_coverage(response, required_keywords)

        scores['response_length'] = self.response_length_score(response)
        scores['context_usage'] = self.context_usage_score(response, contexts)

        # 2. Embedding-Based Metrics (fast, ~10ms)
        scores['answer_relevance'] = self.answer_relevance(query, response)
        scores['context_relevance'] = self.context_relevance(query, contexts)
        scores['context_precision'] = self.context_precision(contexts, response)
        scores['groundedness'] = self.groundedness(response, contexts)

        # 3. LLM-Judge Metrics (optional, slower ~1-2s each)
        if use_llm_judge:
            scores['hallucination'] = 1.0 - self.hallucination_score(response, contexts)
            scores['completeness'] = self.completeness_score(query, response, expected_answer)

        # Calculate aggregate score
        scores['overall_score'] = np.mean(list(scores.values()))

        return EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            scores=scores,
            metadata={
                'num_contexts': len(contexts),
                'response_length': len(response.split()),
                'has_ground_truth': expected_answer is not None
            },
            timestamp=time.time()
        )

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        use_llm_judge: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate multiple test cases.

        Returns aggregate statistics and pass rates.
        """
        results = []

        for case in test_cases:
            result = self.evaluate_all(
                query=case['query'],
                response=case['response'],
                contexts=case['contexts'],
                expected_answer=case.get('expected'),
                required_keywords=case.get('keywords'),
                use_llm_judge=use_llm_judge
            )
            results.append(result)

        # Aggregate statistics
        all_scores = [r.scores for r in results]
        metric_names = list(all_scores[0].keys())

        aggregates = {}
        for metric in metric_names:
            values = [s[metric] for s in all_scores if metric in s]
            aggregates[metric] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'pass_rate': sum(1 for v in values if v >= 0.7) / len(values)
            }

        return {
            'individual_results': [r.to_dict() for r in results],
            'aggregates': aggregates,
            'num_test_cases': len(test_cases),
            'overall_pass_rate': aggregates.get('overall_score', {}).get('pass_rate', 0)
        }


# Example usage
if __name__ == '__main__':
    evaluator = ComprehensiveRAGEvaluator()

    # Test case
    test_case = {
        'query': 'What is the capital of France?',
        'response': 'The capital of France is Paris, which is also its largest city.',
        'contexts': [
            'Paris is the capital and largest city of France.',
            'France is located in Western Europe.'
        ],
        'expected': 'Paris',
        'keywords': ['Paris', 'capital']
    }

    # Evaluate with all metrics
    print("📊 Comprehensive RAG Evaluation")
    print("="*60)

    result = evaluator.evaluate_all(**test_case, use_llm_judge=False)

    print(f"\nQuery: {result.query}")
    print(f"Response: {result.response}\n")

    print("Scores:")
    for metric, score in sorted(result.scores.items()):
        status = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
        print(f"  {status} {metric:.<30} {score:.3f}")

    print(f"\n🎯 Overall Score: {result.scores['overall_score']:.3f}")

    # Batch evaluation
    print("\n" + "="*60)
    print("Batch Evaluation (5 test cases)")
    print("="*60)

    test_cases = [test_case] * 5
    batch_results = evaluator.evaluate_batch(test_cases, use_llm_judge=False)

    print(f"\nTest Cases: {batch_results['num_test_cases']}")
    print(f"Overall Pass Rate: {batch_results['overall_pass_rate']:.1%}\n")

    print("Metric Averages:")
    for metric, stats in batch_results['aggregates'].items():
        if metric != 'overall_score':
            print(f"  {metric:.<30} {stats['mean']:.3f} ± {stats['std']:.3f}")
