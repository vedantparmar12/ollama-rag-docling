"""
TruLens Evaluation for RAG System
Best for: Production monitoring and real-time evaluation

Advantages over RAGAS:
- Real-time monitoring
- Visual dashboard
- LLM-free metrics available
- Production-ready feedback loops
"""

from trulens_eval import Tru, Feedback, TruCustomApp
from trulens_eval.feedback.provider import OpenAI, Huggingface
from trulens_eval.app import App
import numpy as np
from typing import Dict, Any, List


class TruLensRAGEvaluator:
    """
    Comprehensive RAG evaluation using TruLens.

    Metrics:
    1. Context Relevance (LLM-free with embeddings)
    2. Groundedness (checks hallucinations)
    3. Answer Relevance
    4. Latency
    5. Token usage
    """

    def __init__(self, use_ollama: bool = True, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize TruLens evaluator.

        Args:
            use_ollama: Use local Ollama instead of OpenAI
            ollama_base_url: Ollama API endpoint
        """
        self.tru = Tru()
        self.use_ollama = use_ollama

        # Initialize feedback provider
        if use_ollama:
            # Use Hugging Face models (free, local)
            self.provider = Huggingface()
        else:
            # Use OpenAI (requires API key)
            self.provider = OpenAI()

    def setup_feedback_functions(self) -> List[Feedback]:
        """
        Configure evaluation metrics.

        Returns:
            List of TruLens Feedback functions
        """
        feedbacks = []

        # 1. Context Relevance (LLM-free with embeddings)
        f_context_relevance = Feedback(
            self.provider.context_relevance,
            name="Context Relevance"
        ).on_input_output()
        feedbacks.append(f_context_relevance)

        # 2. Groundedness (checks for hallucinations)
        f_groundedness = Feedback(
            self.provider.groundedness_measure_with_cot_reasons,
            name="Groundedness"
        ).on_input_output()
        feedbacks.append(f_groundedness)

        # 3. Answer Relevance
        f_answer_relevance = Feedback(
            self.provider.relevance,
            name="Answer Relevance"
        ).on_input_output()
        feedbacks.append(f_answer_relevance)

        return feedbacks

    def evaluate_rag_response(
        self,
        query: str,
        response: str,
        retrieved_contexts: List[str],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG response.

        Args:
            query: User query
            response: Generated answer
            retrieved_contexts: List of retrieved chunks
            metadata: Optional metadata (latency, tokens, etc.)

        Returns:
            Dictionary of metric scores
        """
        # Create app record
        record = {
            'input': query,
            'output': response,
            'contexts': retrieved_contexts,
            'metadata': metadata or {}
        }

        # Run evaluations
        feedbacks = self.setup_feedback_functions()
        results = {}

        for feedback in feedbacks:
            score = feedback.run(record)
            results[feedback.name] = score

        # Add metadata metrics
        if metadata:
            results['latency_ms'] = metadata.get('latency_ms', 0)
            results['tokens_used'] = metadata.get('tokens_used', 0)

        return results

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of dicts with 'query', 'response', 'contexts'

        Returns:
            Aggregate statistics
        """
        all_results = []

        for case in test_cases:
            result = self.evaluate_rag_response(
                query=case['query'],
                response=case['response'],
                retrieved_contexts=case['contexts'],
                metadata=case.get('metadata')
            )
            all_results.append(result)

        # Aggregate metrics
        metrics = list(all_results[0].keys())
        aggregated = {}

        for metric in metrics:
            values = [r[metric] for r in all_results if metric in r]
            aggregated[metric] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return {
            'individual_results': all_results,
            'aggregate': aggregated,
            'num_test_cases': len(test_cases)
        }

    def start_dashboard(self, port: int = 8501):
        """
        Launch TruLens dashboard for visualization.

        Args:
            port: Dashboard port (default: 8501)
        """
        print(f"🚀 Starting TruLens dashboard on port {port}...")
        print(f"   Access at: http://localhost:{port}")
        self.tru.run_dashboard(port=port)


# Example usage
if __name__ == '__main__':
    evaluator = TruLensRAGEvaluator(use_ollama=True)

    # Test case
    test_case = {
        'query': 'What is the capital of France?',
        'response': 'The capital of France is Paris.',
        'contexts': [
            'Paris is the capital and largest city of France.',
            'France is a country in Western Europe.'
        ],
        'metadata': {
            'latency_ms': 234,
            'tokens_used': 45
        }
    }

    # Evaluate
    results = evaluator.evaluate_rag_response(**test_case)
    print("\n📊 Evaluation Results:")
    for metric, score in results.items():
        print(f"   {metric}: {score:.3f}")

    # Batch evaluation
    test_cases = [test_case] * 10  # 10 test cases
    batch_results = evaluator.evaluate_batch(test_cases)
    print(f"\n📈 Batch Results ({batch_results['num_test_cases']} cases):")
    for metric, stats in batch_results['aggregate'].items():
        print(f"   {metric}:")
        print(f"      Mean: {stats['mean']:.3f}")
        print(f"      Std:  {stats['std']:.3f}")
