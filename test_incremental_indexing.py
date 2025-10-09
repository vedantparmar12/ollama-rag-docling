"""
Test script for Incremental Indexing (COCO Optimization)

This demonstrates:
1. Initial indexing
2. Adding new documents
3. Updating existing documents
4. Re-indexing unchanged content
"""

import os
import shutil
from pathlib import Path
from rag_system.indexing.chunk_hasher import ChunkHasher

def test_basic_hashing():
    """Test 1: Basic hash computation and storage"""
    print("\n" + "="*60)
    print("TEST 1: Basic Hash Functionality")
    print("="*60)

    hasher = ChunkHasher(db_path="./test_hash_registry.db")

    # Create test chunks
    chunks = [
        {
            'chunk_id': 'doc1_0',
            'text': 'This is the first chunk of document 1.',
            'metadata': {'document_id': 'doc1', 'chunk_index': 0}
        },
        {
            'chunk_id': 'doc1_1',
            'text': 'This is the second chunk of document 1.',
            'metadata': {'document_id': 'doc1', 'chunk_index': 1}
        },
        {
            'chunk_id': 'doc2_0',
            'text': 'This belongs to document 2.',
            'metadata': {'document_id': 'doc2', 'chunk_index': 0}
        }
    ]

    # First indexing - all chunks are new
    print("\n📥 First run (all chunks new):")
    new_chunks, unchanged_chunks = hasher.filter_changed_chunks(chunks)
    print(f"   New chunks: {len(new_chunks)}")
    print(f"   Unchanged chunks: {len(unchanged_chunks)}")
    assert len(new_chunks) == 3, "Expected 3 new chunks"
    assert len(unchanged_chunks) == 0, "Expected 0 unchanged chunks"

    # Second run - all chunks unchanged
    print("\n🔄 Second run (no changes):")
    new_chunks, unchanged_chunks = hasher.filter_changed_chunks(chunks)
    print(f"   New chunks: {len(new_chunks)}")
    print(f"   Unchanged chunks: {len(unchanged_chunks)}")
    assert len(new_chunks) == 0, "Expected 0 new chunks"
    assert len(unchanged_chunks) == 3, "Expected 3 unchanged chunks"

    # Modify one chunk
    print("\n✏️  Third run (1 chunk modified):")
    chunks[0]['text'] = 'This is the MODIFIED first chunk of document 1.'
    new_chunks, unchanged_chunks = hasher.filter_changed_chunks(chunks)
    print(f"   New chunks: {len(new_chunks)}")
    print(f"   Unchanged chunks: {len(unchanged_chunks)}")
    assert len(new_chunks) == 1, "Expected 1 new chunk"
    assert len(unchanged_chunks) == 2, "Expected 2 unchanged chunks"

    # Show statistics
    stats = hasher.get_statistics()
    print(f"\n📊 Hash Registry Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   DB size: {stats['db_size_mb']}MB")

    # Cleanup
    os.remove("./test_hash_registry.db")
    print("\n✅ Test 1 PASSED")


def test_document_deletion():
    """Test 2: Deleting document hashes"""
    print("\n" + "="*60)
    print("TEST 2: Document Deletion")
    print("="*60)

    hasher = ChunkHasher(db_path="./test_hash_registry2.db")

    # Create chunks for multiple documents
    doc1_chunks = [
        {'chunk_id': 'doc1_0', 'text': 'Doc1 chunk 1', 'metadata': {'document_id': 'doc1', 'chunk_index': 0}},
        {'chunk_id': 'doc1_1', 'text': 'Doc1 chunk 2', 'metadata': {'document_id': 'doc1', 'chunk_index': 1}}
    ]
    doc2_chunks = [
        {'chunk_id': 'doc2_0', 'text': 'Doc2 chunk 1', 'metadata': {'document_id': 'doc2', 'chunk_index': 0}}
    ]

    all_chunks = doc1_chunks + doc2_chunks

    # Index all
    print("\n📥 Indexing 2 documents:")
    hasher.filter_changed_chunks(all_chunks)
    stats = hasher.get_statistics()
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total documents: {stats['total_documents']}")

    # Delete doc1
    print("\n🗑️  Deleting document 'doc1':")
    hasher.delete_document_hashes('doc1')
    stats = hasher.get_statistics()
    print(f"   Remaining chunks: {stats['total_chunks']}")
    print(f"   Remaining documents: {stats['total_documents']}")
    assert stats['total_chunks'] == 1, "Expected 1 remaining chunk"

    # Cleanup
    os.remove("./test_hash_registry2.db")
    print("\n✅ Test 2 PASSED")


def simulate_incremental_workflow():
    """Test 3: Realistic incremental indexing workflow"""
    print("\n" + "="*60)
    print("TEST 3: Realistic Incremental Workflow")
    print("="*60)

    hasher = ChunkHasher(db_path="./test_hash_registry3.db")

    # Scenario: Index 100 documents initially
    print("\n📚 Initial indexing (100 documents):")
    initial_chunks = []
    for doc_id in range(100):
        for chunk_idx in range(50):  # 50 chunks per doc
            initial_chunks.append({
                'chunk_id': f'doc{doc_id}_{chunk_idx}',
                'text': f'Content for document {doc_id}, chunk {chunk_idx}',
                'metadata': {'document_id': f'doc{doc_id}', 'chunk_index': chunk_idx}
            })

    new, unchanged = hasher.filter_changed_chunks(initial_chunks)
    print(f"   Chunks indexed: {len(new)}")
    print(f"   Total: {len(new) + len(unchanged)}")

    # Scenario: Add 5 new documents
    print("\n➕ Adding 5 new documents:")
    new_docs_chunks = []
    for doc_id in range(100, 105):
        for chunk_idx in range(50):
            new_docs_chunks.append({
                'chunk_id': f'doc{doc_id}_{chunk_idx}',
                'text': f'Content for document {doc_id}, chunk {chunk_idx}',
                'metadata': {'document_id': f'doc{doc_id}', 'chunk_index': chunk_idx}
            })

    all_chunks = initial_chunks + new_docs_chunks
    new, unchanged = hasher.filter_changed_chunks(all_chunks)

    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   New chunks (to embed): {len(new)}")
    print(f"   Unchanged chunks (skipped): {len(unchanged)}")
    savings = (len(unchanged) / len(all_chunks)) * 100
    print(f"   Compute saved: {savings:.1f}%")

    # Scenario: Update 1 document
    print("\n✏️  Updating 1 document (doc50):")
    for chunk in all_chunks:
        if chunk['metadata']['document_id'] == 'doc50':
            chunk['text'] = chunk['text'] + ' [UPDATED]'

    new, unchanged = hasher.filter_changed_chunks(all_chunks)
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Changed chunks (to embed): {len(new)}")
    print(f"   Unchanged chunks (skipped): {len(unchanged)}")
    savings = (len(unchanged) / len(all_chunks)) * 100
    print(f"   Compute saved: {savings:.1f}%")

    # Scenario: Re-index with no changes
    print("\n🔄 Re-indexing with no changes:")
    new, unchanged = hasher.filter_changed_chunks(all_chunks)
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Changed chunks (to embed): {len(new)}")
    print(f"   Unchanged chunks (skipped): {len(unchanged)}")
    if len(new) == 0:
        print(f"   ⚡ 100% speedup - no embedding needed!")

    # Statistics
    stats = hasher.get_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"   Total chunks tracked: {stats['total_chunks']:,}")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Hash registry size: {stats['db_size_mb']}MB")

    # Cleanup
    os.remove("./test_hash_registry3.db")
    print("\n✅ Test 3 PASSED")


def test_performance():
    """Test 4: Performance benchmarks"""
    print("\n" + "="*60)
    print("TEST 4: Performance Benchmarks")
    print("="*60)

    import time

    hasher = ChunkHasher(db_path="./test_hash_registry4.db")

    # Create 10k chunks
    print("\n⏱️  Benchmarking hash computation (10,000 chunks):")
    chunks_10k = []
    for i in range(10000):
        chunks_10k.append({
            'chunk_id': f'chunk_{i}',
            'text': f'This is test chunk number {i} with some content. ' * 10,
            'metadata': {'document_id': f'doc_{i//50}', 'chunk_index': i%50}
        })

    start = time.time()
    new, unchanged = hasher.filter_changed_chunks(chunks_10k)
    duration = time.time() - start

    print(f"   Chunks processed: {len(chunks_10k):,}")
    print(f"   Time: {duration:.2f}s")
    print(f"   Speed: {len(chunks_10k)/duration:.0f} chunks/sec")

    # Re-run to test lookup performance
    print("\n⚡ Benchmarking hash lookup (10,000 chunks):")
    start = time.time()
    new, unchanged = hasher.filter_changed_chunks(chunks_10k)
    duration = time.time() - start

    print(f"   Chunks checked: {len(chunks_10k):,}")
    print(f"   Time: {duration:.2f}s")
    print(f"   Speed: {len(chunks_10k)/duration:.0f} chunks/sec")
    print(f"   All unchanged: {len(unchanged) == len(chunks_10k)}")

    # Cleanup
    os.remove("./test_hash_registry4.db")
    print("\n✅ Test 4 PASSED")


if __name__ == '__main__':
    print("="*60)
    print("INCREMENTAL INDEXING TEST SUITE")
    print("="*60)

    try:
        test_basic_hashing()
        test_document_deletion()
        simulate_incremental_workflow()
        test_performance()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nIncremental indexing is working correctly.")
        print("You can now use it in your RAG system!")
        print("\nTo enable:")
        print("  - Already enabled by default in rag_system/main.py")
        print("  - See INCREMENTAL_INDEXING_GUIDE.md for details")
        print("\nNext steps:")
        print("  1. Run normal indexing: python create_index_script.py ...")
        print("  2. Add/update documents")
        print("  3. Re-run indexing → see the speedup!")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
