"""
DeepEval Evaluation for RAG System
Best for: Unit testing and CI/CD integration

Advantages over RAGAS:
- Pytest integration
- Synthetic test data generation
- Custom metrics easy to add
- LLM-free metrics available
"""

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric
)
from deepeval.test_case import LLMTestCase
from typing import List, Dict, Any
import json


class DeepEvalRAGTester:
    """
    RAG evaluation using DeepEval framework.

    Metrics:
    1. Answer Relevancy - How relevant is the answer to query?
    2. Faithfulness - Is answer grounded in context?
    3. Contextual Precision - Are retrieved docs relevant?
    4. Contextual Recall - Did we retrieve all relevant info?
    5. Hallucination Detection - Is answer making things up?
    """

    def __init__(self, model: str = "gpt-3.5-turbo", threshold: float = 0.7):
        """
        Initialize DeepEval metrics.

        Args:
            model: LLM model for evaluation (or use local Ollama)
            threshold: Minimum score to pass (0.0-1.0)
        """
        self.threshold = threshold
        self.model = model

        # Initialize metrics
        self.metrics = {
            'answer_relevancy': AnswerRelevancyMetric(threshold=threshold, model=model),
            'faithfulness': FaithfulnessMetric(threshold=threshold, model=model),
            'contextual_precision': ContextualPrecisionMetric(threshold=threshold, model=model),
            'hallucination': HallucinationMetric(threshold=threshold, model=model)
        }

    def create_test_case(
        self,
        query: str,
        response: str,
        contexts: List[str],
        expected_output: str = None,
        retrieval_context: List[str] = None
    ) -> LLMTestCase:
        """
        Create a DeepEval test case.

        Args:
            query: User input
            response: RAG system output
            contexts: Retrieved contexts
            expected_output: Ground truth answer (optional)
            retrieval_context: All available contexts (for recall)

        Returns:
            LLMTestCase object
        """
        return LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=expected_output,
            retrieval_context=contexts,
            context=retrieval_context or contexts
        )

    def evaluate_single(
        self,
        test_case: LLMTestCase,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single test case.

        Args:
            test_case: LLMTestCase object
            metrics: List of metric names to evaluate (default: all)

        Returns:
            Dictionary with metric scores and pass/fail
        """
        if metrics is None:
            metrics = list(self.metrics.keys())

        results = {}

        for metric_name in metrics:
            metric = self.metrics[metric_name]
            metric.measure(test_case)

            results[metric_name] = {
                'score': metric.score,
                'threshold': metric.threshold,
                'passed': metric.score >= metric.threshold,
                'reason': metric.reason if hasattr(metric, 'reason') else None
            }

        return results

    def evaluate_batch(
        self,
        test_cases: List[LLMTestCase],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of LLMTestCase objects
            metrics: List of metric names to evaluate

        Returns:
            Aggregate results with pass rates
        """
        all_results = []

        for test_case in test_cases:
            result = self.evaluate_single(test_case, metrics)
            all_results.append(result)

        # Calculate pass rates
        if metrics is None:
            metrics = list(self.metrics.keys())

        pass_rates = {}
        for metric_name in metrics:
            passed = sum(1 for r in all_results if r[metric_name]['passed'])
            pass_rates[metric_name] = {
                'passed': passed,
                'total': len(test_cases),
                'pass_rate': passed / len(test_cases),
                'avg_score': sum(r[metric_name]['score'] for r in all_results) / len(test_cases)
            }

        return {
            'individual_results': all_results,
            'pass_rates': pass_rates,
            'overall_pass_rate': sum(pr['pass_rate'] for pr in pass_rates.values()) / len(pass_rates)
        }

    def run_pytest_suite(self, test_cases: List[LLMTestCase]):
        """
        Run as pytest suite (for CI/CD).

        Usage in test file:
            def test_rag_quality():
                tester = DeepEvalRAGTester()
                tester.run_pytest_suite(test_cases)
        """
        from deepeval import assert_test

        for test_case in test_cases:
            assert_test(test_case, self.metrics.values())

    def generate_synthetic_test_data(
        self,
        documents: List[str],
        num_questions: int = 10
    ) -> List[LLMTestCase]:
        """
        Auto-generate test cases from documents.

        Args:
            documents: List of document texts
            num_questions: Number of QA pairs to generate

        Returns:
            List of synthetic test cases
        """
        from deepeval.synthesizer import Synthesizer

        synthesizer = Synthesizer()
        synthetic_data = synthesizer.generate_goldens(
            contexts=[documents],
            num_goldens=num_questions
        )

        return [
            self.create_test_case(
                query=golden.input,
                response="",  # To be filled by RAG
                contexts=golden.context,
                expected_output=golden.expected_output
            )
            for golden in synthetic_data
        ]


# LLM-FREE Metrics (No API calls needed!)
class LLMFreeMetrics:
    """
    Evaluation metrics that don't require LLM calls.
    Fast, free, and deterministic!
    """

    @staticmethod
    def exact_match(response: str, expected: str) -> float:
        """Exact string match (0 or 1)"""
        return 1.0 if response.strip().lower() == expected.strip().lower() else 0.0

    @staticmethod
    def contains_keywords(response: str, keywords: List[str]) -> float:
        """Check if response contains required keywords"""
        response_lower = response.lower()
        found = sum(1 for kw in keywords if kw.lower() in response_lower)
        return found / len(keywords) if keywords else 0.0

    @staticmethod
    def bleu_score(response: str, reference: str) -> float:
        """BLEU score (n-gram overlap)"""
        from nltk.translate.bleu_score import sentence_bleu
        import nltk
        nltk.download('punkt', quiet=True)

        reference_tokens = nltk.word_tokenize(reference.lower())
        response_tokens = nltk.word_tokenize(response.lower())

        return sentence_bleu([reference_tokens], response_tokens)

    @staticmethod
    def rouge_score(response: str, reference: str) -> Dict[str, float]:
        """ROUGE scores (recall-oriented)"""
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, response)

        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    @staticmethod
    def bert_score(response: str, reference: str) -> float:
        """BERTScore (semantic similarity using embeddings)"""
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn([response], [reference], lang='en', verbose=False)
        return F1.item()

    @staticmethod
    def cosine_similarity(response_embedding, reference_embedding) -> float:
        """Cosine similarity between embeddings"""
        import numpy as np

        return np.dot(response_embedding, reference_embedding) / (
            np.linalg.norm(response_embedding) * np.linalg.norm(reference_embedding)
        )


# Example usage
if __name__ == '__main__':
    tester = DeepEvalRAGTester(threshold=0.7)

    # Create test case
    test_case = tester.create_test_case(
        query="What is the capital of France?",
        response="The capital of France is Paris, a major European city.",
        contexts=[
            "Paris is the capital and largest city of France.",
            "France is located in Western Europe."
        ],
        expected_output="Paris is the capital of France."
    )

    # Evaluate
    results = tester.evaluate_single(test_case)

    print("📊 DeepEval Results:")
    for metric, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"   {metric}: {result['score']:.3f} {status}")

    # LLM-free metrics
    print("\n⚡ LLM-Free Metrics (instant, no cost):")
    print(f"   BLEU: {LLMFreeMetrics.bleu_score(test_case.actual_output, test_case.expected_output):.3f}")
    print(f"   Exact Match: {LLMFreeMetrics.exact_match(test_case.actual_output, test_case.expected_output):.3f}")
    print(f"   Keywords: {LLMFreeMetrics.contains_keywords(test_case.actual_output, ['Paris', 'capital']):.3f}")
