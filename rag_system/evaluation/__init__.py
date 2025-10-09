"""
RAG Evaluation Framework

Provides multiple evaluation methods for RAG systems:
1. ComprehensiveRAGEvaluator - All-in-one solution (recommended)
2. TruLensRAGEvaluator - Production monitoring with dashboard
3. DeepEvalRAGTester - Unit testing and CI/CD integration
4. UpTrainRAGEvaluator - Free evaluation with open-source models
"""

from .rag_evaluator import ComprehensiveRAGEvaluator, EvaluationResult

# Optional imports (only if packages are installed)
try:
    from .trulens_eval import TruLensRAGEvaluator
except ImportError:
    TruLensRAGEvaluator = None

try:
    from .deepeval_metrics import DeepEvalRAGTester, LLMFreeMetrics
except ImportError:
    DeepEvalRAGTester = None
    LLMFreeMetrics = None

try:
    from .uptrain_eval import UpTrainRAGEvaluator
except ImportError:
    UpTrainRAGEvaluator = None


__all__ = [
    'ComprehensiveRAGEvaluator',
    'EvaluationResult',
    'TruLensRAGEvaluator',
    'DeepEvalRAGTester',
    'LLMFreeMetrics',
    'UpTrainRAGEvaluator'
]


def get_evaluator(framework: str = 'comprehensive', **kwargs):
    """
    Factory function to get the appropriate evaluator.

    Args:
        framework: One of 'comprehensive', 'trulens', 'deepeval', 'uptrain'
        **kwargs: Framework-specific arguments

    Returns:
        Evaluator instance

    Examples:
        >>> evaluator = get_evaluator('comprehensive')
        >>> evaluator = get_evaluator('trulens', use_ollama=True)
        >>> evaluator = get_evaluator('deepeval', threshold=0.8)
    """
    framework = framework.lower()

    if framework == 'comprehensive':
        return ComprehensiveRAGEvaluator(**kwargs)

    elif framework == 'trulens':
        if TruLensRAGEvaluator is None:
            raise ImportError("TruLens not installed. Run: pip install trulens-eval")
        return TruLensRAGEvaluator(**kwargs)

    elif framework == 'deepeval':
        if DeepEvalRAGTester is None:
            raise ImportError("DeepEval not installed. Run: pip install deepeval")
        return DeepEvalRAGTester(**kwargs)

    elif framework == 'uptrain':
        if UpTrainRAGEvaluator is None:
            raise ImportError("UpTrain not installed. Run: pip install uptrain")
        return UpTrainRAGEvaluator(**kwargs)

    else:
        raise ValueError(f"Unknown framework: {framework}. Choose from: comprehensive, trulens, deepeval, uptrain")
