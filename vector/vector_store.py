"""
Vector Store for Metric Embeddings
Uses FAISS for efficient similarity search
"""

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
from dataclasses import dataclass, asdict

from ..models.metric_data import MetricIdentifier, MetricFeatures


@dataclass
class MetricEmbedding:
    """Stored metric embedding with metadata"""
    metric_id: str  # Unique identifier
    metric: MetricIdentifier
    embedding: List[float]
    features_summary: str  # Text representation
    timestamp: float  # When stored


class VectorStore:
    """
    FAISS-based vector store for metric embeddings
    
    Features:
    - Fast similarity search using FAISS
    - Persistent storage to disk
    - Metadata storage alongside vectors
    - Efficient batch operations
    """
    
    def __init__(
        self,
        dimension: int = 768,
        storage_path: str = "vector_store"
    ):
        """
        Initialize vector store
        
        Args:
            dimension: Embedding dimension (default: 768 for nomic-embed-text)
            storage_path: Path to store index and metadata
        """
        self.dimension = dimension
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        
        # Metadata storage (index -> metadata)
        self.metadata: Dict[int, MetricEmbedding] = {}
        self.metric_id_to_index: Dict[str, int] = {}
        
        # Load existing index if available
        self._load_index()
        
        logger.info(
            f"Vector store initialized: {self.index.ntotal} vectors, "
            f"dim={dimension}"
        )
    
    def add(
        self,
        metric_id: str,
        metric: MetricIdentifier,
        embedding: List[float],
        features: MetricFeatures,
        timestamp: float
    ) -> int:
        """
        Add a metric embedding to the store
        
        Args:
            metric_id: Unique identifier for metric
            metric: MetricIdentifier
            embedding: Embedding vector
            features: MetricFeatures for text representation
            timestamp: Unix timestamp
        
        Returns:
            Index in vector store
        """
        try:
            # Check if already exists
            if metric_id in self.metric_id_to_index:
                logger.warning(f"Metric {metric_id} already exists, updating...")
                return self.update(metric_id, embedding, features, timestamp)
            
            # Convert to numpy array
            vector = np.array([embedding], dtype=np.float32)
            
            # Add to FAISS index
            self.index.add(vector)
            idx = self.index.ntotal - 1
            
            # Store metadata
            metadata = MetricEmbedding(
                metric_id=metric_id,
                metric=metric,
                embedding=embedding,
                features_summary=features.to_text(),
                timestamp=timestamp
            )
            self.metadata[idx] = metadata
            self.metric_id_to_index[metric_id] = idx
            
            logger.debug(f"Added metric {metric_id} at index {idx}")
            return idx
            
        except Exception as e:
            logger.error(f"Error adding to vector store: {e}")
            raise
    
    def update(
        self,
        metric_id: str,
        embedding: List[float],
        features: MetricFeatures,
        timestamp: float
    ) -> int:
        """
        Update an existing metric embedding
        
        Args:
            metric_id: Metric identifier
            embedding: New embedding vector
            features: New features
            timestamp: Update timestamp
        
        Returns:
            Index in vector store
        """
        try:
            if metric_id not in self.metric_id_to_index:
                raise ValueError(f"Metric {metric_id} not found")
            
            idx = self.metric_id_to_index[metric_id]
            
            # Update metadata
            old_metadata = self.metadata[idx]
            self.metadata[idx] = MetricEmbedding(
                metric_id=metric_id,
                metric=old_metadata.metric,
                embedding=embedding,
                features_summary=features.to_text(),
                timestamp=timestamp
            )
            
            # Note: FAISS doesn't support in-place updates
            # For true updates, rebuild index or use IndexIDMap
            logger.debug(f"Updated metadata for {metric_id}")
            
            return idx
            
        except Exception as e:
            logger.error(f"Error updating vector store: {e}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filters: Optional[Dict[str, str]] = None
    ) -> List[Tuple[MetricEmbedding, float]]:
        """
        Search for similar metrics
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Optional label filters
        
        Returns:
            List of (MetricEmbedding, distance) tuples
        
        Example:
            >>> results = store.search(embedding, k=10)
            >>> for metadata, distance in results:
            ...     print(f"{metadata.metric.name}: {distance:.3f}")
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("Vector store is empty")
                return []
            
            # Convert to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search FAISS index
            distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            
            # Gather results
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for missing results
                    continue
                
                metadata = self.metadata.get(idx)
                if not metadata:
                    continue
                
                # Apply filters if provided
                if filters:
                    match = all(
                        metadata.metric.labels.get(key) == value
                        for key, value in filters.items()
                    )
                    if not match:
                        continue
                
                results.append((metadata, float(distance)))
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def save(self):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            index_path = self.storage_path / "faiss.index"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = self.storage_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'metric_id_to_index': self.metric_id_to_index
                }, f)
            
            logger.info(
                f"Saved vector store: {self.index.ntotal} vectors to {self.storage_path}"
            )
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def _load_index(self):
        """Load index and metadata from disk"""
        try:
            index_path = self.storage_path / "faiss.index"
            metadata_path = self.storage_path / "metadata.pkl"
            
            if not index_path.exists() or not metadata_path.exists():
                logger.info("No existing index found, starting fresh")
                return
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.metric_id_to_index = data['metric_id_to_index']
            
            logger.info(
                f"Loaded vector store: {self.index.ntotal} vectors from {self.storage_path}"
            )
            
        except Exception as e:
            logger.warning(f"Could not load index: {e}, starting fresh")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}
            self.metric_id_to_index = {}
    
    def clear(self):
        """Clear all data from vector store"""
        self.index.reset()
        self.metadata.clear()
        self.metric_id_to_index.clear()
        logger.info("Vector store cleared")
    
    def stats(self) -> Dict[str, any]:
        """Get vector store statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "unique_metrics": len(self.metric_id_to_index),
            "storage_path": str(self.storage_path)
        }
