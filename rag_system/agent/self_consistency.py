"""Self-Consistency Module for Critical Query Verification

Implements self-consistency checking by generating multiple answers
and selecting the most consistent one. Helps reduce hallucinations
on critical or ambiguous queries.

Based on research: "Improving the Reliability of LLMs: Combining CoT, RAG,
Self-Consistency, and Self-Verification" (arXiv:2505.09031)
"""

import asyncio
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SelfConsistencyChecker:
    """
    Generates multiple answers and selects the most consistent one.

    This reduces hallucinations by:
    1. Generating N diverse answers (varying temperature)
    2. Computing pairwise similarity between answers
    3. Selecting the answer with highest average similarity to others
    4. Flagging low-consistency answers as potentially unreliable
    """

    def __init__(
        self,
        embedding_model=None,
        n_samples: int = 5,
        temperature: float = 0.7,
        consistency_threshold: float = 0.75
    ):
        """
        Initialize self-consistency checker.

        Args:
            embedding_model: Model for computing answer embeddings
            n_samples: Number of diverse answers to generate
            temperature: Temperature for diverse generation
            consistency_threshold: Minimum consistency score to pass
        """
        self.embedding_model = embedding_model
        self.n_samples = n_samples
        self.temperature = temperature
        self.consistency_threshold = consistency_threshold

    def _lazy_load_embedding_model(self):
        """Lazily load embedding model if not provided."""
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            print("🔧 Loading embedding model for self-consistency checking...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded.")
        return self.embedding_model

    async def generate_with_consistency(
        self,
        generate_fn,
        query: str,
        context: str,
        **gen_kwargs
    ) -> Dict[str, Any]:
        """
        Generate answer with self-consistency checking.

        Args:
            generate_fn: Function to generate answers (async)
            query: User query
            context: Retrieved context
            **gen_kwargs: Additional generation arguments

        Returns:
            Dict with:
                - answer: Most consistent answer
                - consistency_score: Average similarity score
                - warning: Optional warning if consistency is low
                - all_answers: All generated answers (for debugging)
        """
        # Generate N diverse answers
        print(f"🔄 Generating {self.n_samples} diverse answers for consistency check...")

        tasks = []
        for i in range(self.n_samples):
            # Vary temperature slightly for diversity
            temp = self.temperature + (i - self.n_samples // 2) * 0.1
            temp = max(0.3, min(1.0, temp))  # Clamp to [0.3, 1.0]

            task = generate_fn(
                query=query,
                context=context,
                temperature=temp,
                **gen_kwargs
            )
            tasks.append(task)

        # Run in parallel
        answers = await asyncio.gather(*tasks)

        # Extract text from answers (handle different return types)
        answer_texts = []
        for ans in answers:
            if isinstance(ans, dict):
                answer_texts.append(ans.get('answer', ans.get('response', str(ans))))
            else:
                answer_texts.append(str(ans))

        # Compute consistency
        consistency_result = self._compute_consistency(answer_texts)

        # Select most consistent answer
        best_idx = consistency_result['best_answer_idx']
        best_answer = answer_texts[best_idx]
        consistency_score = consistency_result['consistency_scores'][best_idx]

        result = {
            'answer': best_answer,
            'consistency_score': consistency_score,
            'all_answers': answer_texts,
            'similarity_matrix': consistency_result['similarity_matrix'].tolist()
        }

        # Add warning if consistency is low
        if consistency_score < self.consistency_threshold:
            result['warning'] = (
                f"Low consistency detected (score: {consistency_score:.3f}). "
                f"This answer may be unreliable. Please verify manually or rephrase your query."
            )
            print(f"⚠️ {result['warning']}")
        else:
            print(f"✅ High consistency (score: {consistency_score:.3f})")

        return result

    def _compute_consistency(self, answers: List[str]) -> Dict[str, Any]:
        """
        Compute consistency scores across multiple answers.

        Args:
            answers: List of generated answers

        Returns:
            Dict with similarity matrix and consistency scores
        """
        # Load embedding model if needed
        model = self._lazy_load_embedding_model()

        # Generate embeddings for all answers
        embeddings = model.encode(answers, show_progress_bar=False)

        # Compute pairwise similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Compute consistency score for each answer
        # (average similarity to all other answers)
        consistency_scores = []
        for i in range(len(answers)):
            # Exclude self-similarity (always 1.0)
            others_sim = np.concatenate([
                similarity_matrix[i, :i],
                similarity_matrix[i, i+1:]
            ])
            avg_similarity = np.mean(others_sim)
            consistency_scores.append(avg_similarity)

        # Find most consistent answer
        best_idx = np.argmax(consistency_scores)

        return {
            'similarity_matrix': similarity_matrix,
            'consistency_scores': consistency_scores,
            'best_answer_idx': best_idx,
            'mean_consistency': np.mean(consistency_scores),
            'std_consistency': np.std(consistency_scores)
        }

    def check_consistency_sync(self, answers: List[str]) -> Dict[str, Any]:
        """
        Synchronous version: Check consistency of pre-generated answers.

        Args:
            answers: List of answers to check

        Returns:
            Consistency analysis results
        """
        consistency_result = self._compute_consistency(answers)

        best_idx = consistency_result['best_answer_idx']
        consistency_score = consistency_result['consistency_scores'][best_idx]

        result = {
            'best_answer': answers[best_idx],
            'best_answer_idx': best_idx,
            'consistency_score': consistency_score,
            'all_consistency_scores': consistency_result['consistency_scores'],
            'mean_consistency': consistency_result['mean_consistency'],
            'similarity_matrix': consistency_result['similarity_matrix'].tolist()
        }

        if consistency_score < self.consistency_threshold:
            result['warning'] = (
                f"Low consistency ({consistency_score:.3f}). "
                "Answers vary significantly - verification recommended."
            )

        return result


# Example usage and testing
if __name__ == "__main__":
    # Test with sample answers
    sample_answers = [
        "The capital of France is Paris.",
        "Paris is the capital city of France.",
        "France's capital is Paris, located in the north of the country.",
        "The capital is Paris.",
        "Berlin is the capital of Germany."  # Inconsistent answer
    ]

    checker = SelfConsistencyChecker(n_samples=5, consistency_threshold=0.75)
    result = checker.check_consistency_sync(sample_answers)

    print("\n" + "="*60)
    print("SELF-CONSISTENCY CHECK RESULTS")
    print("="*60)
    print(f"Best Answer: {result['best_answer']}")
    print(f"Consistency Score: {result['consistency_score']:.3f}")
    print(f"Mean Consistency: {result['mean_consistency']:.3f}")

    if 'warning' in result:
        print(f"\n⚠️ WARNING: {result['warning']}")
    else:
        print(f"\n✅ High consistency - answer is reliable")

    print(f"\nAll Consistency Scores:")
    for i, score in enumerate(result['all_consistency_scores']):
        print(f"  Answer {i+1}: {score:.3f} - {sample_answers[i][:50]}...")
