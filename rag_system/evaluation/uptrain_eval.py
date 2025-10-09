"""
UpTrain Evaluation for RAG System
Best for: Cost-effective evaluation with open-source models

Advantages over RAGAS:
- 100% free (uses open-source models)
- No API keys required
- Fast and efficient
- Data-driven insights
"""

from uptrain import EvalLLM, Evals
from uptrain.framework import Settings
from typing import List, Dict, Any
import json


class UpTrainRAGEvaluator:
    """
    RAG evaluation using UpTrain (open-source, free).

    Available Metrics:
    1. Context Relevance
    2. Factual Accuracy
    3. Response Completeness
    4. Response Conciseness
    5. Response Relevance
    6. Critique Language (tone analysis)
    7. Guideline Adherence (custom rules)
    """

    def __init__(self, use_local_model: bool = True):
        """
        Initialize UpTrain evaluator.

        Args:
            use_local_model: Use local open-source model (free)
                           False = use OpenAI (requires API key)
        """
        if use_local_model:
            # Use free open-source models
            settings = Settings(
                model="ollama/llama3.2",  # Use your local Ollama
                ollama_api_base="http://localhost:11434"
            )
        else:
            settings = None  # Will use OpenAI

        self.eval_llm = EvalLLM(settings=settings)

    def evaluate_rag_response(
        self,
        query: str,
        response: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG response.

        Args:
            query: User question
            response: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Expected answer (optional)

        Returns:
            Evaluation scores
        """
        # Prepare data for UpTrain
        data = [{
            "question": query,
            "response": response,
            "context": "\n".join(contexts),
            "ground_truth": ground_truth
        }]

        # Define evaluation metrics
        checks = [
            Evals.CONTEXT_RELEVANCE,
            Evals.FACTUAL_ACCURACY,
            Evals.RESPONSE_RELEVANCE,
            Evals.RESPONSE_COMPLETENESS,
            Evals.RESPONSE_CONCISENESS
        ]

        # Run evaluation
        results = self.eval_llm.evaluate(
            data=data,
            checks=checks
        )

        return results[0] if results else {}

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of dicts with query, response, contexts

        Returns:
            List of evaluation results
        """
        # Prepare batch data
        data = [
            {
                "question": case['query'],
                "response": case['response'],
                "context": "\n".join(case['contexts']),
                "ground_truth": case.get('ground_truth')
            }
            for case in test_cases
        ]

        # Define checks
        checks = [
            Evals.CONTEXT_RELEVANCE,
            Evals.FACTUAL_ACCURACY,
            Evals.RESPONSE_RELEVANCE,
            Evals.RESPONSE_COMPLETENESS
        ]

        # Batch evaluation
        results = self.eval_llm.evaluate(
            data=data,
            checks=checks
        )

        return results

    def evaluate_with_guidelines(
        self,
        query: str,
        response: str,
        guidelines: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate response against custom guidelines.

        Args:
            query: User question
            response: Generated answer
            guidelines: List of rules to check

        Example guidelines:
        - "Response must be under 100 words"
        - "Must cite sources"
        - "Must not contain medical advice"

        Returns:
            Guideline adherence scores
        """
        data = [{
            "question": query,
            "response": response
        }]

        # Create custom guideline checks
        checks = []
        for guideline in guidelines:
            checks.append(Evals.GUIDELINE_ADHERENCE(guideline=guideline))

        results = self.eval_llm.evaluate(
            data=data,
            checks=checks
        )

        return results[0] if results else {}

    def analyze_tone(
        self,
        response: str,
        desired_tone: str = "professional"
    ) -> Dict[str, Any]:
        """
        Analyze response tone/language.

        Args:
            response: Generated answer
            desired_tone: Expected tone (professional, friendly, concise, etc.)

        Returns:
            Tone analysis results
        """
        data = [{"response": response}]

        checks = [
            Evals.CRITIQUE_LANGUAGE,
            Evals.RESPONSE_TONE(desired_tone=desired_tone)
        ]

        results = self.eval_llm.evaluate(
            data=data,
            checks=checks
        )

        return results[0] if results else {}

    def compare_responses(
        self,
        query: str,
        response_a: str,
        response_b: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        A/B test two different responses.

        Args:
            query: User question
            response_a: First response
            response_b: Second response
            contexts: Retrieved contexts

        Returns:
            Comparison results showing which is better
        """
        # Evaluate both
        result_a = self.evaluate_rag_response(query, response_a, contexts)
        result_b = self.evaluate_rag_response(query, response_b, contexts)

        # Compare
        comparison = {
            'response_a': result_a,
            'response_b': result_b,
            'winner': {}
        }

        # Determine winner for each metric
        for metric in result_a.keys():
            if metric in result_b:
                score_a = result_a[metric].get('score', 0)
                score_b = result_b[metric].get('score', 0)

                if score_a > score_b:
                    comparison['winner'][metric] = 'A'
                elif score_b > score_a:
                    comparison['winner'][metric] = 'B'
                else:
                    comparison['winner'][metric] = 'Tie'

        return comparison


# Example usage
if __name__ == '__main__':
    # Initialize with local model (free!)
    evaluator = UpTrainRAGEvaluator(use_local_model=True)

    # Test case
    test_case = {
        'query': 'What is the capital of France?',
        'response': 'The capital of France is Paris, which is also the largest city in the country.',
        'contexts': [
            'Paris is the capital and largest city of France.',
            'France is a country in Western Europe.'
        ],
        'ground_truth': 'Paris'
    }

    # Evaluate
    print("📊 UpTrain Evaluation (FREE, no API keys!):")
    results = evaluator.evaluate_rag_response(**test_case)

    for metric, score in results.items():
        print(f"   {metric}: {score}")

    # Custom guidelines
    print("\n📋 Guideline Adherence Check:")
    guideline_results = evaluator.evaluate_with_guidelines(
        query=test_case['query'],
        response=test_case['response'],
        guidelines=[
            "Response must mention the city name",
            "Response should be under 50 words",
            "Response must be factually correct"
        ]
    )

    for guideline, result in guideline_results.items():
        print(f"   {guideline}: {result}")

    # Tone analysis
    print("\n🎭 Tone Analysis:")
    tone_results = evaluator.analyze_tone(
        response=test_case['response'],
        desired_tone="professional"
    )
    print(json.dumps(tone_results, indent=2))
