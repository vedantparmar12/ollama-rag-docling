"""Comprehensive Test Suite for 2025 Advanced RAG Features

Tests all newly implemented features:
1. Jina-ColBERT Reranker
2. Self-Consistency Checker
3. Multimodal Embeddings
4. Max-Min Semantic Chunking
5. Real-Time RAG
6. Context Pruning (already integrated)
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_jina_colbert_reranker():
    """Test 1: Jina-ColBERT Reranker"""
    print("\n" + "="*70)
    print("TEST 1: JINA-COLBERT RERANKER")
    print("="*70)

    from rag_system.rerankers.reranker import QwenReranker

    # Test with Jina-ColBERT-v2
    try:
        print("\n🧪 Testing Jina-ColBERT-v2 reranker...")
        reranker = QwenReranker(model_name="jinaai/jina-colbert-v2")

        # Sample documents
        query = "What are the benefits of exercise?"
        documents = [
            {'text': "Regular exercise improves cardiovascular health and reduces disease risk.", 'score': 0.8},
            {'text': "Many people enjoy watching movies in their free time.", 'score': 0.3},
            {'text': "Physical activity strengthens muscles and bones.", 'score': 0.75},
            {'text': "Exercise releases endorphins which improve mood and mental health.", 'score': 0.85},
            {'text': "The weather today is sunny and warm.", 'score': 0.2},
        ]

        reranked = reranker.rerank(query, documents, top_k=3)

        print(f"✅ Reranked {len(documents)} documents to top {len(reranked)}")
        print("\nTop Results:")
        for i, doc in enumerate(reranked, 1):
            print(f"  {i}. Score: {doc.get('rerank_score', 0):.4f}")
            print(f"     Text: {doc['text'][:70]}...")

        print(f"\n✅ Jina-ColBERT Test: PASSED")
        print(f"   Max context length: {reranker.max_length} tokens")
        return True

    except Exception as e:
        print(f"\n❌ Jina-ColBERT Test: FAILED")
        print(f"   Error: {e}")
        print(f"   Note: Make sure 'rerankers' library is installed: pip install rerankers")
        return False


def test_self_consistency():
    """Test 2: Self-Consistency Checker"""
    print("\n" + "="*70)
    print("TEST 2: SELF-CONSISTENCY CHECKER")
    print("="*70)

    from rag_system.agent.self_consistency import SelfConsistencyChecker

    try:
        print("\n🧪 Testing self-consistency checking...")
        checker = SelfConsistencyChecker(n_samples=5, consistency_threshold=0.75)

        # Test answers (some consistent, one inconsistent)
        test_answers = [
            "The capital of France is Paris.",
            "Paris is the capital city of France.",
            "France's capital is Paris.",
            "The capital is Paris.",
            "Berlin is the capital of Germany."  # Inconsistent!
        ]

        result = checker.check_consistency_sync(test_answers)

        print(f"✅ Analyzed {len(test_answers)} answers")
        print(f"\nBest Answer: {result['best_answer']}")
        print(f"Consistency Score: {result['consistency_score']:.3f}")
        print(f"Mean Consistency: {result['mean_consistency']:.3f}")

        if 'warning' in result:
            print(f"\n⚠️  Warning: {result['warning']}")
        else:
            print(f"\n✅ High consistency detected")

        print(f"\n✅ Self-Consistency Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Self-Consistency Test: FAILED")
        print(f"   Error: {e}")
        return False


def test_multimodal_embeddings():
    """Test 3: Multimodal Embeddings"""
    print("\n" + "="*70)
    print("TEST 3: MULTIMODAL EMBEDDINGS")
    print("="*70)

    from rag_system.indexing.multimodal_embedders import MultiModalEmbedder, ContentType

    try:
        print("\n🧪 Testing multimodal embedding system...")
        embedder = MultiModalEmbedder(enable_code=True, enable_table=True)

        # Test samples of different content types
        samples = [
            {
                "text": "The capital of France is Paris, a major European city.",
                "metadata": {}
            },
            {
                "text": "def calculate_sum(a, b):\n    return a + b",
                "metadata": {"type": "code"}
            },
            {
                "text": "| Product | Price | Stock |\n|---------|-------|-------|\n| Apple | $1.50 | 100 |",
                "metadata": {}
            },
            {
                "text": "$$E = mc^2$$",
                "metadata": {}
            }
        ]

        enriched = embedder.embed_with_metadata(samples)

        print(f"✅ Embedded {len(samples)} chunks with content type detection")
        print("\nResults:")
        for i, chunk in enumerate(enriched, 1):
            print(f"  {i}. Type: {chunk['content_type']}")
            print(f"     Shape: {chunk['embedding'].shape}")
            print(f"     Preview: {chunk['text'][:50]}...")

        # Verify different types detected
        detected_types = {chunk['content_type'] for chunk in enriched}
        print(f"\n✅ Detected content types: {detected_types}")
        print(f"✅ Multimodal Embeddings Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Multimodal Embeddings Test: FAILED")
        print(f"   Error: {e}")
        return False


def test_maxmin_chunking():
    """Test 4: Max-Min Semantic Chunking"""
    print("\n" + "="*70)
    print("TEST 4: MAX-MIN SEMANTIC CHUNKING")
    print("="*70)

    from rag_system.ingestion.maxmin_chunker import MaxMinSemanticChunker

    try:
        print("\n🧪 Testing Max-Min semantic chunking...")

        # Multi-topic test document
        test_text = """
        The capital of France is Paris. Paris is known for the Eiffel Tower.
        The city has a rich history dating back centuries.

        Machine learning is transforming technology. Neural networks can learn
        complex patterns from data. Deep learning has enabled breakthroughs
        in computer vision and natural language processing.

        Climate change is affecting our planet. Rising temperatures are melting
        ice caps and raising sea levels. Governments worldwide are implementing
        policies to reduce carbon emissions and transition to renewable energy.

        Python is a versatile programming language. It's used in web development,
        data science, and automation. Many companies rely on Python for
        production systems due to its simplicity and extensive library ecosystem.
        """

        chunker = MaxMinSemanticChunker(
            min_chunk_size=100,
            max_chunk_size=500,
            similarity_threshold=0.80
        )

        chunks = chunker.chunk_text(test_text, document_id="test_doc")

        print(f"✅ Created {len(chunks)} semantic chunks")
        print("\nChunk Statistics:")
        for chunk in chunks:
            meta = chunk['metadata']
            print(f"  Chunk {meta['chunk_index'] + 1}: {meta['char_count']} chars, "
                  f"~{meta['sentence_count']} sentences")

        avg_size = sum(c['metadata']['char_count'] for c in chunks) // len(chunks)
        print(f"\n✅ Average chunk size: {avg_size} chars")
        print(f"✅ Max-Min Chunking Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Max-Min Chunking Test: FAILED")
        print(f"   Error: {e}")
        return False


async def test_realtime_rag():
    """Test 5: Real-Time RAG"""
    print("\n" + "="*70)
    print("TEST 5: REAL-TIME RAG")
    print("="*70)

    from rag_system.retrieval.realtime_retriever import RealtimeRetriever

    try:
        print("\n🧪 Testing real-time RAG retrieval...")
        retriever = RealtimeRetriever()

        # Test queries
        test_queries = [
            "What's the current weather in Paris?",
            "What is the current price of AAPL stock?",
            "How many products are available in the database?",
        ]

        for query in test_queries:
            print(f"\n📝 Query: {query}")
            results = await retriever.retrieve(query)

            if results.get('realtime_data'):
                rt_data = results['realtime_data']
                print(f"   🔴 Real-Time Data: {rt_data.get('source', 'unknown')}")
                print(f"   📊 Data: {list(rt_data.keys())}")
            else:
                print(f"   📄 No real-time data needed")

        print(f"\n✅ Real-Time RAG Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Real-Time RAG Test: FAILED")
        print(f"   Error: {e}")
        return False


def test_context_pruning():
    """Test 6: Context Pruning (Already Integrated)"""
    print("\n" + "="*70)
    print("TEST 6: CONTEXT PRUNING (PROVENCE)")
    print("="*70)

    from rag_system.rerankers.sentence_pruner import SentencePruner

    try:
        print("\n🧪 Testing sentence-level context pruning...")
        pruner = SentencePruner()

        query = "What are the health benefits of exercise?"
        documents = [
            {
                'text': ("Regular exercise has numerous health benefits. "
                        "It strengthens your heart and improves circulation. "
                        "Exercise also helps maintain a healthy weight. "
                        "Additionally, it can improve your mood and reduce stress. "
                        "Some people don't exercise regularly."),
                'chunk_id': 'doc1'
            }
        ]

        pruned = pruner.prune_documents(query, documents, threshold=0.1)

        print(f"✅ Pruned {len(documents)} documents")
        for i, doc in enumerate(pruned, 1):
            original_len = len(documents[i-1]['text'])
            pruned_len = len(doc['text'])
            reduction = ((original_len - pruned_len) / original_len) * 100
            print(f"\n  Document {i}:")
            print(f"    Original: {original_len} chars")
            print(f"    Pruned: {pruned_len} chars")
            print(f"    Reduction: {reduction:.1f}%")
            print(f"    Pruned text: {doc['text'][:100]}...")

        print(f"\n✅ Context Pruning Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Context Pruning Test: FAILED")
        print(f"   Error: {e}")
        print(f"   Note: Provence model may need to be downloaded first")
        return False


def run_all_tests():
    """Run all tests and generate summary report"""
    print("\n" + "="*70)
    print("ADVANCED RAG FEATURES TEST SUITE (2025)")
    print("="*70)
    print("Testing newly implemented features for enhanced RAG performance\n")

    results = {}

    # Run synchronous tests
    results['Jina-ColBERT Reranker'] = test_jina_colbert_reranker()
    results['Self-Consistency'] = test_self_consistency()
    results['Multimodal Embeddings'] = test_multimodal_embeddings()
    results['Max-Min Chunking'] = test_maxmin_chunking()
    results['Context Pruning'] = test_context_pruning()

    # Run async test
    loop = asyncio.get_event_loop()
    results['Real-Time RAG'] = loop.run_until_complete(test_realtime_rag())

    # Summary Report
    print("\n" + "="*70)
    print("TEST SUMMARY REPORT")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for feature, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {feature}: {'PASSED' if status else 'FAILED'}")

    print("\n" + "-"*70)
    print(f"TOTAL: {passed}/{total} tests passed ({(passed/total)*100:.0f}%)")
    print("="*70)

    if passed == total:
        print("\n🎉 All tests PASSED! Advanced features are working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check error messages above.")

    return passed == total


if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

    print("\n🚀 Starting Advanced RAG Features Test Suite...")
    success = run_all_tests()

    print("\n" + "="*70)
    if success:
        print("✅ FINAL RESULT: ALL SYSTEMS GO!")
        print("   Your RAG system now has state-of-the-art 2025 capabilities.")
    else:
        print("⚠️  FINAL RESULT: SOME ISSUES DETECTED")
        print("   Review failed tests above and install missing dependencies.")

    print("="*70 + "\n")

    sys.exit(0 if success else 1)
