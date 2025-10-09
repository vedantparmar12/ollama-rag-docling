"""
Content-Aware Chunk Hasher for Incremental Indexing
Implements COCO (Content-aware Optimization) indexing to avoid re-embedding unchanged chunks.
"""

import hashlib
import json
import sqlite3
from typing import List, Dict, Any, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ChunkHasher:
    """
    Manages chunk hashing and tracking to enable incremental indexing.

    Features:
    - SHA-256 hashing of chunk content
    - SQLite-based hash registry
    - Change detection for documents
    - Batch hash lookups
    """

    def __init__(self, db_path: str = "./index_store/hash_registry.db"):
        """
        Initialize the chunk hasher.

        Args:
            db_path: Path to SQLite database for hash registry
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create hash registry database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_hashes (
                chunk_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_id
            ON chunk_hashes(document_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_hash
            ON chunk_hashes(content_hash)
        """)

        conn.commit()
        conn.close()
        logger.info(f"Hash registry initialized at {self.db_path}")

    def compute_hash(self, chunk: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of chunk content.

        Args:
            chunk: Chunk dictionary with 'text' field

        Returns:
            Hexadecimal hash string
        """
        # Hash the actual text content
        content = chunk.get('text', '')

        # Optionally include metadata for stricter change detection
        # Uncomment if you want metadata changes to trigger re-embedding:
        # metadata = chunk.get('metadata', {})
        # content += json.dumps(metadata, sort_keys=True)

        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def get_existing_hashes(self, chunk_ids: List[str]) -> Dict[str, str]:
        """
        Retrieve existing hashes for given chunk IDs.

        Args:
            chunk_ids: List of chunk IDs to lookup

        Returns:
            Dictionary mapping chunk_id to content_hash
        """
        if not chunk_ids:
            return {}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(chunk_ids))
        cursor.execute(
            f"SELECT chunk_id, content_hash FROM chunk_hashes WHERE chunk_id IN ({placeholders})",
            chunk_ids
        )

        result = dict(cursor.fetchall())
        conn.close()

        return result

    def get_document_hashes(self, document_id: str) -> Dict[str, str]:
        """
        Get all chunk hashes for a specific document.

        Args:
            document_id: Document identifier

        Returns:
            Dictionary mapping chunk_id to content_hash
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT chunk_id, content_hash FROM chunk_hashes WHERE document_id = ?",
            (document_id,)
        )

        result = dict(cursor.fetchall())
        conn.close()

        return result

    def store_hashes(self, chunks: List[Dict[str, Any]], hashes: List[str]):
        """
        Store chunk hashes in the registry.

        Args:
            chunks: List of chunk dictionaries
            hashes: List of computed hashes (same order as chunks)
        """
        if len(chunks) != len(hashes):
            raise ValueError("Chunks and hashes lists must have same length")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        data = []
        for chunk, hash_val in zip(chunks, hashes):
            doc_id = chunk.get('metadata', {}).get('document_id', 'unknown')
            chunk_idx = chunk.get('metadata', {}).get('chunk_index', -1)

            data.append((
                chunk['chunk_id'],
                hash_val,
                doc_id,
                chunk_idx
            ))

        cursor.executemany("""
            INSERT OR REPLACE INTO chunk_hashes
            (chunk_id, content_hash, document_id, chunk_index)
            VALUES (?, ?, ?, ?)
        """, data)

        conn.commit()
        conn.close()
        logger.info(f"Stored {len(data)} chunk hashes")

    def filter_changed_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Separate new/changed chunks from unchanged chunks.

        Args:
            chunks: List of all chunks to process

        Returns:
            Tuple of (new_chunks, unchanged_chunks)
        """
        chunk_ids = [c['chunk_id'] for c in chunks]
        existing_hashes = self.get_existing_hashes(chunk_ids)

        new_chunks = []
        unchanged_chunks = []
        new_hashes = []

        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            current_hash = self.compute_hash(chunk)

            if chunk_id in existing_hashes:
                if existing_hashes[chunk_id] == current_hash:
                    # Content unchanged - skip embedding
                    unchanged_chunks.append(chunk)
                else:
                    # Content changed - needs re-embedding
                    new_chunks.append(chunk)
                    new_hashes.append(current_hash)
            else:
                # New chunk - needs embedding
                new_chunks.append(chunk)
                new_hashes.append(current_hash)

        # Store new hashes
        if new_chunks:
            self.store_hashes(new_chunks, new_hashes)

        return new_chunks, unchanged_chunks

    def delete_document_hashes(self, document_id: str):
        """
        Remove all hash entries for a document.

        Args:
            document_id: Document identifier
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM chunk_hashes WHERE document_id = ?", (document_id,))
        deleted = cursor.rowcount

        conn.commit()
        conn.close()
        logger.info(f"Deleted {deleted} hash entries for document {document_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with total_chunks, total_documents, db_size
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM chunk_hashes")
        total_chunks = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT document_id) FROM chunk_hashes")
        total_documents = cursor.fetchone()[0]

        conn.close()

        db_size_mb = self.db_path.stat().st_size / (1024 * 1024)

        return {
            'total_chunks': total_chunks,
            'total_documents': total_documents,
            'db_size_mb': round(db_size_mb, 2)
        }


# Example usage
if __name__ == '__main__':
    hasher = ChunkHasher()

    # Test chunks
    chunks = [
        {
            'chunk_id': 'doc1_0',
            'text': 'The quick brown fox jumps over the lazy dog.',
            'metadata': {'document_id': 'doc1', 'chunk_index': 0}
        },
        {
            'chunk_id': 'doc1_1',
            'text': 'Lorem ipsum dolor sit amet.',
            'metadata': {'document_id': 'doc1', 'chunk_index': 1}
        }
    ]

    # First run - all chunks are new
    new, unchanged = hasher.filter_changed_chunks(chunks)
    print(f"Run 1: {len(new)} new, {len(unchanged)} unchanged")

    # Second run - all chunks unchanged
    new, unchanged = hasher.filter_changed_chunks(chunks)
    print(f"Run 2: {len(new)} new, {len(unchanged)} unchanged")

    # Modify one chunk
    chunks[0]['text'] = 'The quick brown fox jumps VERY HIGH.'
    new, unchanged = hasher.filter_changed_chunks(chunks)
    print(f"Run 3 (1 modified): {len(new)} new, {len(unchanged)} unchanged")

    # Stats
    print(f"\nStatistics: {hasher.get_statistics()}")
