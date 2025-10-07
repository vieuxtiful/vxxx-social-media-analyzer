"""
SocialMediaSentimentAnalyzer - Advanced Opinion Mining Tool
>>Word2World Social<<

This tool leverages sophisticated mathematical frameworks to provide advanced sentiment analysis with uncertainty
quantification and explainable AI capabilities.
>>bridging ideas together<<

Key Features:
- Bayesian-SVM hybrid classification with confidence scores
- Deep learning (CNN/LSTM) for sequential pattern analysis
- Explainable AI for transparent sentiment predictions
- Multilingual processing for global social media analysis
- Intelligent caching for real-time streaming data analysis

Author: vieuxtiful
License: MIT
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import heapq
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading

class ScalableSemanticIndex:
    """
    Advanced semantic indexing system that combines multiple ANN strategies
    for different scale and accuracy requirements.
    """
    
    def __init__(self, 
                 dimension: int = 128,
                 initial_capacity: int = 100000,
                 max_capacity: int = 10000000,
                 ann_method: str = 'hnsw',  # 'hnsw', 'ivf', 'lsh', 'composite'
                 precision_level: str = 'adaptive'):  # 'high', 'medium', 'adaptive'
        
        self.dimension = dimension
        self.current_capacity = initial_capacity
        self.max_capacity = max_capacity
        self.ann_method = ann_method
        self.precision_level = precision_level
        
        # Multiple indexing strategies
        self.primary_index = None
        self.secondary_index = None
        self.fallback_index = None
        
        # Query optimization
        self.query_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance monitoring
        self.query_times = []
        self.index_update_times = []
        
        self._initialize_indices()
        
    def _initialize_indices(self):
        """Initialize appropriate ANN indices based on configuration."""
        try:
            if self.ann_method in ['hnsw', 'composite']:
                import hnswlib
                self.primary_index = hnswlib.Index(space='cosine', dim=self.dimension)
                self.primary_index.init_index(max_elements=self.current_capacity, 
                                            ef_construction=200, M=16)
                self.primary_index.set_ef(50)  # Query time accuracy/speed trade-off
                
            if self.ann_method in ['ivf', 'composite']:
                try:
                    import faiss
                    self.secondary_index = faiss.IndexIVFFlat(
                        faiss.IndexFlatIP(self.dimension), 
                        self.dimension, 
                        min(16384, self.current_capacity // 39)
                    )
                    # Train with random data initially
                    training_data = np.random.random((1000, self.dimension)).astype('float32')
                    self.secondary_index.train(training_data)
                except ImportError:
                    print("FAISS not available, falling back to HNSW only")
                    
            if self.ann_method == 'composite':
                # Composite index uses both HNSW and IVF for different query patterns
                self.query_router = QueryRouter()
                
        except ImportError as e:
            print(f"ANN library import failed: {e}. Using fallback methods.")
            self._setup_fallback_indices()
    
    def _setup_fallback_indices(self):
        """Setup fallback indexing methods when optimized libraries aren't available."""
        print("Using fallback BallTree indexing - consider installing hnswlib/faiss for better performance")
        from sklearn.neighbors import BallTree
        self.fallback_index = BallTree(metric='cosine')
        self.fallback_data = None
        
    def add_items(self, vectors: np.ndarray, ids: List[str], incremental: bool = True):
        """
        Add vectors to the index with efficient batch processing.
        
        Args:
            vectors: Array of shape (n_vectors, dimension)
            ids: List of string identifiers
            incremental: Whether to update indices incrementally
        """
        if len(vectors) == 0:
            return
            
        start_time = time.time()
        
        # Ensure vectors are normalized for cosine similarity
        vectors = self._normalize_vectors(vectors)
        
        if self.primary_index is not None and hasattr(self.primary_index, 'add_items'):
            try:
                # Convert to float32 for HNSW
                vectors_float32 = vectors.astype(np.float32)
                self.primary_index.add_items(vectors_float32, np.arange(len(ids)))
            except Exception as e:
                print(f"HNSW add failed: {e}. Rebuilding index...")
                self._rebuild_primary_index(vectors, ids)
                
        if self.secondary_index is not None:
            try:
                vectors_float32 = vectors.astype(np.float32)
                self.secondary_index.add(vectors_float32)
            except Exception as e:
                print(f"FAISS add failed: {e}")
                
        if self.fallback_index is not None:
            if self.fallback_data is None:
                self.fallback_data = vectors
                self.fallback_ids = ids
            else:
                self.fallback_data = np.vstack([self.fallback_data, vectors])
                self.fallback_ids.extend(ids)
            self.fallback_index = BallTree(self.fallback_data, metric='cosine')
            
        # Update capacity tracking
        self.current_capacity += len(vectors)
        
        # Clear query cache since index changed
        self.query_cache.clear()
        
        self.index_update_times.append(time.time() - start_time)
        
    def query(self, 
              query_vector: np.ndarray, 
              k: int = 10,
              precision: Optional[str] = None,
              timeout_ms: int = 100) -> Tuple[List[str], List[float]]:
        """
        Query for nearest neighbors with configurable precision/timeout.
        
        Args:
            query_vector: Query vector of shape (dimension,)
            k: Number of neighbors to return
            precision: Override default precision level
            timeout_ms: Maximum query time in milliseconds
            
        Returns:
            Tuple of (neighbor_ids, distances)
        """
        cache_key = (tuple(query_vector), k, precision)
        if cache_key in self.query_cache:
            self.cache_hits += 1
            return self.query_cache[cache_key]
            
        self.cache_misses += 1
        start_time = time.time()
        
        query_vector = self._normalize_vectors(query_vector.reshape(1, -1))[0]
        actual_precision = precision or self.precision_level
        
        results = None
        
        # Adaptive query routing based on precision requirements
        if actual_precision == 'high' or self.ann_method != 'composite':
            results = self._query_high_precision(query_vector, k, timeout_ms)
        elif actual_precision == 'medium':
            results = self._query_medium_precision(query_vector, k, timeout_ms)
        else:  # adaptive
            results = self._query_adaptive(query_vector, k, timeout_ms)
            
        query_time = time.time() - start_time
        self.query_times.append(query_time)
        
        # Cache results for similar future queries
        if query_time < timeout_ms / 1000.0:  # Only cache if within timeout
            self.query_cache[cache_key] = results
            # Limit cache size
            if len(self.query_cache) > 10000:
                self._prune_cache()
                
        return results
    
    def _query_high_precision(self, query_vector: np.ndarray, k: int, timeout_ms: int):
        """High precision query using multiple verification steps."""
        # Use primary HNSW index
        if self.primary_index is not None:
            try:
                query_vector_float32 = query_vector.astype(np.float32).reshape(1, -1)
                indices, distances = self.primary_index.knn_query(query_vector_float32, k=k)
                # HNSW returns squared distances, convert to cosine distances
                cosine_distances = 1 - (1 - distances[0]) / 2  # Convert to [0, 1] range
                return [str(i) for i in indices[0]], cosine_distances.tolist()
            except Exception as e:
                print(f"HNSW query failed: {e}")
                
        return self._query_fallback(query_vector, k)
    
    def _query_medium_precision(self, query_vector: np.ndarray, k: int, timeout_ms: int):
        """Medium precision query balancing speed and accuracy."""
        # Try FAISS IVF first for speed
        if self.secondary_index is not None:
            try:
                query_vector_float32 = query_vector.astype(np.float32).reshape(1, -1)
                distances, indices = self.secondary_index.search(query_vector_float32, k)
                return [str(i) for i in indices[0]], distances[0].tolist()
            except Exception as e:
                print(f"FAISS query failed: {e}")
                
        return self._query_high_precision(query_vector, k, timeout_ms)
    
    def _query_adaptive(self, query_vector: np.ndarray, k: int, timeout_ms: int):
        """Adaptive query that starts fast and refines if time permits."""
        start_time = time.time()
        
        # First, quick FAISS query
        if self.secondary_index is not None:
            try:
                query_vector_float32 = query_vector.astype(np.float32).reshape(1, -1)
                distances, indices = self.secondary_index.search(query_vector_float32, k * 2)  # Get extra candidates
                candidate_ids = [str(i) for i in indices[0]]
                candidate_distances = distances[0].tolist()
                
                # If we have time, refine with HNSW
                elapsed = (time.time() - start_time) * 1000
                if elapsed < timeout_ms / 2 and self.primary_index is not None:
                    refined_ids, refined_distances = self._query_high_precision(query_vector, k, timeout_ms - elapsed)
                    if len(refined_ids) > 0:
                        return refined_ids, refined_distances
                        
                return candidate_ids[:k], candidate_distances[:k]
            except Exception as e:
                print(f"Adaptive query failed: {e}")
                
        return self._query_high_precision(query_vector, k, timeout_ms)
    
    def _query_fallback(self, query_vector: np.ndarray, k: int):
        """Fallback query using BallTree."""
        if self.fallback_index is not None and self.fallback_data is not None:
            distances, indices = self.fallback_index.query([query_vector], k=k)
            neighbor_ids = [self.fallback_ids[i] for i in indices[0]]
            return neighbor_ids, distances[0].tolist()
        return [], []
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def _rebuild_primary_index(self, vectors: np.ndarray, ids: List[str]):
        """Rebuild primary index when it becomes corrupted or inefficient."""
        print("Rebuilding primary index...")
        import hnswlib
        
        self.primary_index = hnswlib.Index(space='cosine', dim=self.dimension)
        self.primary_index.init_index(max_elements=max(self.current_capacity, len(vectors) * 2), 
                                    ef_construction=200, M=16)
        
        if len(vectors) > 0:
            vectors_float32 = vectors.astype(np.float32)
            self.primary_index.add_items(vectors_float32, np.arange(len(ids)))
        
        self.primary_index.set_ef(50)
    
    def _prune_cache(self):
        """Prune query cache using LRU strategy."""
        # Simple strategy: remove oldest 20%
        keys = list(self.query_cache.keys())
        remove_count = len(keys) // 5
        for key in keys[:remove_count]:
            del self.query_cache[key]
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for monitoring."""
        query_times = np.array(self.query_times[-1000:] if self.query_times else [0])
        update_times = np.array(self.index_update_times[-100:] if self.index_update_times else [0])
        
        return {
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'avg_query_time_ms': np.mean(query_times) * 1000,
            'p95_query_time_ms': np.percentile(query_times, 95) * 1000,
            'avg_update_time_ms': np.mean(update_times) * 1000,
            'current_capacity': self.current_capacity,
            'cache_size': len(self.query_cache),
            'index_method': self.ann_method
        }

class DistributedSemanticEngine:
    """
    Distributed version for massive-scale platforms.
    Uses sharding and parallel processing.
    """
    
    def __init__(self, 
                 num_shards: int = 8,
                 shard_dimension: int = 32,  # Reduced dimension per shard
                 coordinator_url: Optional[str] = None):
        
        self.num_shards = num_shards
        self.shard_dimension = shard_dimension
        self.shards = []
        self.coordinator_url = coordinator_url
        
        # Initialize shards
        for i in range(num_shards):
            shard = ScalableSemanticIndex(
                dimension=shard_dimension,
                ann_method='hnsw',
                precision_level='adaptive'
            )
            self.shards.append(shard)
            
        # Query coordinator
        self.query_executor = ThreadPoolExecutor(max_workers=num_shards)
        
    def add_items_distributed(self, vectors: np.ndarray, ids: List[str]):
        """Add items using dimensional sharding."""
        if vectors.shape[1] != self.shard_dimension * self.num_shards:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match shard configuration")
            
        # Split vectors across dimensional shards
        shard_vectors = []
        for i in range(self.num_shards):
            start_idx = i * self.shard_dimension
            end_idx = (i + 1) * self.shard_dimension
            shard_vec = vectors[:, start_idx:end_idx]
            shard_vectors.append(shard_vec)
            
        # Add to shards in parallel
        futures = []
        for i, shard in enumerate(self.shards):
            future = self.query_executor.submit(shard.add_items, shard_vectors[i], ids)
            futures.append(future)
            
        # Wait for completion
        for future in futures:
            future.result()
    
    def query_distributed(self, 
                         query_vector: np.ndarray, 
                         k: int = 10,
                         fusion_method: str = 'weighted') -> Tuple[List[str], List[float]]:
        """
        Query across all shards and fuse results.
        
        Args:
            fusion_method: 'weighted', 'borda', 'reciprocal'
        """
        # Split query vector
        shard_queries = []
        for i in range(self.num_shards):
            start_idx = i * self.shard_dimension
            end_idx = (i + 1) * self.shard_dimension
            shard_query = query_vector[start_idx:end_idx]
            shard_queries.append(shard_query)
            
        # Query all shards in parallel
        futures = []
        for i, shard in enumerate(self.shards):
            future = self.query_executor.submit(shard.query, shard_queries[i], k * 2)  # Get more candidates
            futures.append((i, future))
            
        # Collect results
        shard_results = []
        for shard_id, future in futures:
            try:
                ids, distances = future.result(timeout=1.0)  # 1 second timeout per shard
                shard_results.append((shard_id, ids, distances))
            except Exception as e:
                print(f"Shard {shard_id} query failed: {e}")
                shard_results.append((shard_id, [], []))
                
        # Fuse results
        return self._fuse_results(shard_results, k, fusion_method)
    
    def _fuse_results(self, 
                     shard_results: List[Tuple[int, List[str], List[float]]], 
                     k: int,
                     fusion_method: str) -> Tuple[List[str], List[float]]:
        """Fuse results from multiple shards."""
        if fusion_method == 'weighted':
            return self._fuse_weighted(shard_results, k)
        elif fusion_method == 'borda':
            return self._fuse_borda(shard_results, k)
        else:  # reciprocal
            return self._fuse_reciprocal(shard_results, k)
    
    def _fuse_weighted(self, shard_results, k):
        """Weighted fusion based on shard reliability."""
        candidate_scores = defaultdict(float)
        
        for shard_id, ids, distances in shard_results:
            if not ids:
                continue
                
            # Convert distances to similarities (higher is better)
            similarities = [1 - d for d in distances]
            max_sim = max(similarities) if similarities else 1
            
            for i, (item_id, sim) in enumerate(zip(ids, similarities)):
                # Weight by rank position and shard reliability
                rank_weight = 1.0 / (i + 1)  # Higher rank = more weight
                shard_weight = 1.0  # Could be based on shard performance history
                normalized_sim = sim / max_sim
                
                score = rank_weight * shard_weight * normalized_sim
                candidate_scores[item_id] += score
                
        # Get top k candidates
        top_candidates = heapq.nlargest(k, candidate_scores.items(), key=lambda x: x[1])
        top_ids = [candidate[0] for candidate in top_candidates]
        top_scores = [candidate[1] for candidate in top_candidates]
        
        return top_ids, top_scores
    
    def _fuse_borda(self, shard_results, k):
        """Borda count fusion."""
        candidate_ranks = defaultdict(list)
        
        for shard_id, ids, distances in shard_results:
            for rank, item_id in enumerate(ids):
                candidate_ranks[item_id].append(rank)
                
        # Average rank for each candidate
        candidate_avg_ranks = {}
        for item_id, ranks in candidate_ranks.items():
            candidate_avg_ranks[item_id] = sum(ranks) / len(ranks)
            
        # Get top k by average rank (lower is better)
        top_candidates = heapq.nsmallest(k, candidate_avg_ranks.items(), key=lambda x: x[1])
        top_ids = [candidate[0] for candidate in top_candidates]
        top_scores = [1.0 / (candidate[1] + 1) for candidate in top_candidates]  # Convert back to score
        
        return top_ids, top_scores

# Enhanced Word2WorldEngine with scalable indexing
class ScalableWord2WorldEngine(Word2WorldEngine):
    """
    Scalable version of Word2WorldEngine for massive datasets.
    """
    
    def __init__(self, 
                 semantic_analyzer,
                 bridge_threshold: float = 0.3,
                 enable_distributed: bool = False,
                 shard_config: Optional[Dict] = None):
        
        super().__init__(semantic_analyzer, bridge_threshold)
        
        self.enable_distributed = enable_distributed
        self.shard_config = shard_config or {'num_shards': 4, 'shard_dimension': 32}
        
        if enable_distributed and len(self.user_coordinates) > 100000:
            self.semantic_index = DistributedSemanticEngine(**self.shard_config)
        else:
            self.semantic_index = ScalableSemanticIndex(
                dimension=128,  # STAP output dimension
                ann_method='composite',
                precision_level='adaptive'
            )
            
        # Batch processing for efficiency
        self.pending_updates = []
        self.batch_size = 1000
        self.last_batch_process = time.time()
        
    def update_user_semantic_profile_batch(self, user_updates: List[Tuple[str, np.ndarray]]):
        """
        Batch update for better performance with large datasets.
        """
        self.pending_updates.extend(user_updates)
        
        # Process batch if size threshold or time threshold reached
        if (len(self.pending_updates) >= self.batch_size or 
            time.time() - self.last_batch_process > 300):  # 5 minutes
            
            self._process_batch_updates()
            
    def _process_batch_updates(self):
        """Process accumulated batch updates."""
        if not self.pending_updates:
            return
            
        user_ids = [update[0] for update in self.pending_updates]
        vectors = np.array([update[1] for update in self.pending_updates])
        
        # Update semantic index
        self.semantic_index.add_items(vectors, user_ids)
        
        # Update local coordinates (for small-scale operations)
        for user_id, vector in self.pending_updates:
            if user_id in self.user_coordinates:
                # Update existing (simplified - in practice, you'd want more sophisticated merging)
                old = self.user_coordinates[user_id]
                self.user_coordinates[user_id] = SemanticCoordinate(
                    user_id=user_id,
                    coordinate=vector,
                    confidence=min(1.0, old.confidence + 0.1),
                    last_updated=time.time(),
                    post_count=old.post_count + 1
                )
            else:
                self.user_coordinates[user_id] = SemanticCoordinate(
                    user_id=user_id,
                    coordinate=vector,
                    confidence=0.7,
                    last_updated=time.time(),
                    post_count=1
                )
                
        self.pending_updates.clear()
        self.last_batch_process = time.time()
        
    def generate_bridge_recommendations_scalable(self, 
                                               user_id: str, 
                                               max_recommendations: int = 3,
                                               timeout_ms: int = 500) -> List[BridgeRecommendation]:
        """
        Scalable version of bridge recommendation generation.
        """
        if user_id not in self.user_coordinates:
            return []
            
        user_coord = self.user_coordinates[user_id]
        
        # Use scalable semantic index
        neighbor_ids, distances = self.semantic_index.query(
            user_coord.coordinate, 
            k=max_recommendations * 10,  # Get more candidates for filtering
            timeout_ms=timeout_ms
        )
        
        recommendations = []
        for neighbor_id, distance in zip(neighbor_ids, distances):
            if (distance < 0.1 or distance > self.bridge_threshold or 
                neighbor_id == user_id):
                continue
                
            # Same bridge generation logic as before, but scalable
            bridge_content = self._find_bridging_content_scalable(user_id, neighbor_id)
            if bridge_content:
                bridge_strength = self._calculate_bridge_strength(user_coord.coordinate, 
                                                                self.user_coordinates[neighbor_id].coordinate)
                
                explanation = self._generate_bridge_explanation(user_id, neighbor_id, bridge_content)
                
                recommendations.append(BridgeRecommendation(
                    source_user=user_id,
                    target_user=neighbor_id,
                    content_to_share=bridge_content,
                    semantic_distance=distance,
                    bridge_strength=bridge_strength,
                    explanation=explanation
                ))
                
            if len(recommendations) >= max_recommendations:
                break
                
        return sorted(recommendations, key=lambda x: x.bridge_strength, reverse=True)

# Performance monitoring and optimization
class PerformanceOptimizer:
    """
    Continuously optimizes system performance based on usage patterns.
    """
    
    def __init__(self, semantic_engine: ScalableWord2WorldEngine):
        self.engine = semantic_engine
        self.performance_history = []
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
    def _optimization_loop(self):
        """Continuous optimization loop."""
        while True:
            time.sleep(300)  # Run every 5 minutes
            
            try:
                self._analyze_and_optimize()
            except Exception as e:
                print(f"Optimization error: {e}")
                
    def _analyze_and_optimize(self):
        """Analyze performance and apply optimizations."""
        stats = self.engine.semantic_index.get_performance_stats()
        self.performance_history.append(stats)
        
        # Adaptive optimization strategies
        if stats['p95_query_time_ms'] > 100:  # Slow queries
            self._optimize_for_speed()
            
        if stats['cache_hit_rate'] < 0.3:  # Poor cache performance
            self._optimize_caching()
            
        if len(self.engine.user_coordinates) > 500000:  # Large dataset
            self._enable_distributed_if_needed()
    
    def _optimize_for_speed(self):
        """Optimize for query speed."""
        index = self.engine.semantic_index
        if hasattr(index, 'primary_index') and index.primary_index is not None:
            # Reduce HNSW accuracy for speed
            index.primary_index.set_ef(30)  # Lower ef = faster, less accurate
            
    def _optimize_caching(self):
        """Optimize caching strategy."""
        # Implement more sophisticated caching
        pass
        
    def _enable_distributed_if_needed(self):
        """Enable distributed processing if scale requires it."""
        if (not self.engine.enable_distributed and 
            len(self.engine.user_coordinates) > 1000000):
            
            print("Enabling distributed processing for scale...")
            # In practice, you'd migrate to distributed setup
            pass
