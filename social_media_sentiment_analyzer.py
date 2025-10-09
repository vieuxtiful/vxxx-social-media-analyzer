#!/usr/bin/env python3
"""
World2World - Social Media Platform
Bridging Ideas Together

This platform leverages sophisticated mathematical frameworks and advanced opinion mining capabilities to provide advanced sentiment analysis for 
"bridging" similar ideas of distant origins.

Key Features:
- Clear separation of concerns with well-defined classes
- Database abstraction for easy backend switching
- RESTful API ready for React frontend integration
- Scalable design that can grow with your platform
- Comprehensive feedback loop for continuous improvement

Author: vieuxtiful
License: MIT
"""

### Word2World Core Engine ### 
### Bridging Ideas Together ###
Modular Semantic Bridging System
# word2world_core.py

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import logging
from enum import Enum
import uuid

# Database abstraction
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    POST = "post"
    COMMENT = "comment"
    BIO = "bio"
    CAPTION = "caption"
    HASHTAG = "hashtag"
    SHARED_ARTICLE = "shared_article"

class InteractionType(Enum):
    LIKE = "like"
    SHARE = "share"
    COMMENT = "comment"
    SAVE = "save"
    CLICK = "click"
    FOLLOW = "follow"

@dataclass
class UserContent:
    """Unified content model across platform."""
    content_id: str
    user_id: str
    content_type: ContentType
    text: str
    timestamp: datetime
    language: str = "en"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.content_id:
            self.content_id = str(uuid.uuid4())

@dataclass
class UserInteraction:
    """Standardized interaction model."""
    interaction_id: str
    user_id: str
    content_id: str
    interaction_type: InteractionType
    timestamp: datetime
    intensity: float = 1.0  # 1.0 for basic, >1.0 for strong engagement
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.interaction_id:
            self.interaction_id = str(uuid.uuid4())

@dataclass
class SemanticCoordinate:
    """STAP-generated semantic position."""
    user_id: str
    coordinate: np.ndarray
    confidence: float
    last_updated: datetime
    semantic_version: str = "1.0"
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        data = asdict(self)
        data['coordinate'] = self.coordinate.tolist()
        return data

@dataclass
class ContentRecommendation:
    """Structured recommendation output."""
    recommendation_id: str
    user_id: str
    content_id: str
    bridge_strength: float
    semantic_distance: float
    explanation: str
    recommendation_type: str  # 'bridge', 'reinforce', 'explore'
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.recommendation_id:
            self.recommendation_id = str(uuid.uuid4())

# Database Abstraction Layer
class DatabaseInterface(ABC):
    """Abstract base class for database operations."""
    
    @abstractmethod
    def save_user_content(self, content: UserContent) -> bool:
        pass
    
    @abstractmethod
    def get_user_content(self, user_id: str, limit: int = 1000) -> List[UserContent]:
        pass
    
    @abstractmethod
    def save_interaction(self, interaction: UserInteraction) -> bool:
        pass
    
    @abstractmethod
    def get_user_interactions(self, user_id: str, days: int = 30) -> List[UserInteraction]:
        pass
    
    @abstractmethod
    def save_semantic_coordinate(self, coordinate: SemanticCoordinate) -> bool:
        pass
    
    @abstractmethod
    def get_semantic_coordinate(self, user_id: str) -> Optional[SemanticCoordinate]:
        pass
    
    @abstractmethod
    def save_recommendation(self, recommendation: ContentRecommendation) -> bool:
        pass
    
    @abstractmethod
    def get_recommendation_feedback(self, recommendation_id: str) -> Optional[Dict]:
        pass

# Example PostgreSQL Implementation
class PostgreSQLDatabase(DatabaseInterface):
    """PostgreSQL implementation of database interface."""
    
    def __init__(self, connection_string: str):
        # In practice: import psycopg2, sqlalchemy, etc.
        self.connection_string = connection_string
        logger.info("Initialized PostgreSQL database interface")
    
    def save_user_content(self, content: UserContent) -> bool:
        try:
            # Implementation would use psycopg2/sqlalchemy
            logger.info(f"Saved content {content.content_id} for user {content.user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            return False
    
    def get_user_content(self, user_id: str, limit: int = 1000) -> List[UserContent]:
        # Mock implementation - would query database
        return []
    
    def save_interaction(self, interaction: UserInteraction) -> bool:
        logger.info(f"Saved interaction {interaction.interaction_id}")
        return True
    
    def get_user_interactions(self, user_id: str, days: int = 30) -> List[UserInteraction]:
        return []
    
    def save_semantic_coordinate(self, coordinate: SemanticCoordinate) -> bool:
        logger.info(f"Saved semantic coordinate for user {coordinate.user_id}")
        return True
    
    def get_semantic_coordinate(self, user_id: str) -> Optional[SemanticCoordinate]:
        return None
    
    def save_recommendation(self, recommendation: ContentRecommendation) -> bool:
        logger.info(f"Saved recommendation {recommendation.recommendation_id}")
        return True
    
    def get_recommendation_feedback(self, recommendation_id: str) -> Optional[Dict]:
        return None

"""CLASS: User Data Repository
Manages user content and interactions"""
# user_data_repository.py

from word2world_core import *

class UserDataRepository:
    """
    Central repository for user data management.
    Handles storage, retrieval, and preprocessing of user content and interactions.
    """
    
    def __init__(self, database: DatabaseInterface):
        self.database = database
        self.content_cache = {}  # In-memory cache for frequent access
        self.interaction_cache = {}
        
    def add_user_content(self, content: UserContent) -> bool:
        """Add new user content to repository."""
        try:
            # Preprocess content
            processed_content = self._preprocess_content(content)
            
            # Store in database
            success = self.database.save_user_content(processed_content)
            
            if success:
                # Update cache
                if content.user_id not in self.content_cache:
                    self.content_cache[content.user_id] = []
                self.content_cache[content.user_id].append(processed_content)
                
                # Limit cache size
                if len(self.content_cache[content.user_id]) > 1000:
                    self.content_cache[content.user_id] = self.content_cache[content.user_id][-500:]
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding user content: {e}")
            return False
    
    def add_user_interaction(self, interaction: UserInteraction) -> bool:
        """Add new user interaction to repository."""
        try:
            success = self.database.save_interaction(interaction)
            
            if success:
                # Update cache
                cache_key = f"{interaction.user_id}_{interaction.interaction_type.value}"
                if cache_key not in self.interaction_cache:
                    self.interaction_cache[cache_key] = []
                self.interaction_cache[cache_key].append(interaction)
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding user interaction: {e}")
            return False
    
    def get_user_semantic_corpus(self, user_id: str) -> List[str]:
        """Retrieve all textual content for a user as a semantic corpus."""
        # Try cache first
        cached_content = self.content_cache.get(user_id, [])
        
        if not cached_content:
            # Fetch from database
            cached_content = self.database.get_user_content(user_id, limit=1000)
            self.content_cache[user_id] = cached_content
        
        # Extract and clean text
        corpus = []
        for content in cached_content:
            cleaned_text = self._clean_text_for_semantics(content.text)
            if cleaned_text and len(cleaned_text.split()) >= 2:  # Minimum 2 words
                corpus.append(cleaned_text)
        
        return corpus
    
    def get_user_engagement_patterns(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Analyze user engagement patterns for semantic insights."""
        interactions = self.database.get_user_interactions(user_id, days)
        
        if not interactions:
            return {}
        
        # Analyze interaction types
        interaction_counts = {}
        for interaction in interactions:
            key = interaction.interaction_type.value
            interaction_counts[key] = interaction_counts.get(key, 0) + 1
        
        # Calculate engagement intensity
        total_engagement = sum(interaction_counts.values())
        engagement_intensity = sum(
            interaction.intensity for interaction in interactions
        ) / len(interactions)
        
        # Identify preferred content types (via interactions)
        # This would require joining with content data in practice
        
        return {
            'interaction_counts': interaction_counts,
            'total_engagement': total_engagement,
            'engagement_intensity': engagement_intensity,
            'interaction_timeline': self._analyze_temporal_patterns(interactions),
            'preferred_content_types': self._infer_content_preferences(interactions)
        }
    
    def _preprocess_content(self, content: UserContent) -> UserContent:
        """Preprocess content before storage."""
        # Clean text
        content.text = self._clean_text_for_storage(content.text)
        
        # Extract and store hashtags in metadata
        hashtags = self._extract_hashtags(content.text)
        if hashtags:
            content.metadata['hashtags'] = hashtags
        
        # Detect language if not provided
        if not content.language or content.language == 'unknown':
            content.language = self._detect_language(content.text)
        
        return content
    
    def _clean_text_for_storage(self, text: str) -> str:
        """Basic text cleaning for storage."""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Basic profanity filtering (would be more sophisticated in production)
        profanity_patterns = []  # Would contain actual patterns
        for pattern in profanity_patterns:
            text = re.sub(pattern, '[REDACTED]', text, flags=re.IGNORECASE)
        
        return text
    
    def _clean_text_for_semantics(self, text: str) -> str:
        """Clean text specifically for semantic analysis."""
        import re
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove user mentions but keep text
        text = re.sub(r'@(\w+)', r'\1', text)
        
        # Convert hashtags to normal text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        return text.strip()
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        import re
        return re.findall(r'#(\w+)', text)
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # In practice, use langdetect or similar
        if not text or len(text) < 10:
            return 'unknown'
        
        # Simple heuristic based on common words
        english_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have'}
        words = set(text.lower().split())
        
        if any(word in english_words for word in words):
            return 'en'
        
        return 'unknown'
    
    def _analyze_temporal_patterns(self, interactions: List[UserInteraction]) -> Dict:
        """Analyze when users are most active."""
        if not interactions:
            return {}
        
        # Group by hour of day
        hourly_engagement = {}
        for interaction in interactions:
            hour = interaction.timestamp.hour
            hourly_engagement[hour] = hourly_engagement.get(hour, 0) + 1
        
        return {
            'hourly_engagement': hourly_engagement,
            'peak_engagement_hour': max(hourly_engagement, key=hourly_engagement.get) if hourly_engagement else None
        }
    
    def _infer_content_preferences(self, interactions: List[UserInteraction]) -> List[str]:
        """Infer content preferences from interaction patterns."""
        # This would analyze which types of content get the most engagement
        # For now, return empty list - would be implemented with actual content analysis
        return []

"""
CLASS: STAP Pre-Processing Layer
Converts user data to semantic coordinates
"""
# stap_preprocessing.py

from word2world_core import *
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class STAPPreprocessingLayer:
    """
    Transforms user content and interactions into STAP-compatible semantic coordinates.
    """
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = 384  # Standard for all-MiniLM-L6-v2
        
        # STAP parameters (would be loaded from config)
        self.stap_params = {
            'attractive_force_weight': 1.0,
            'repulsive_force_weight': 0.8,
            'learning_rate': 0.01,
            'epochs': 100,
            'negative_samples': 5
        }
        
        # Cache for user embeddings
        self.user_embeddings_cache = {}
        
    def generate_semantic_coordinate(self, 
                                   user_id: str,
                                   content_corpus: List[str],
                                   engagement_patterns: Dict[str, Any]) -> SemanticCoordinate:
        """
        Generate semantic coordinate from user data using STAP principles.
        """
        try:
            # Step 1: Generate base embeddings from text corpus
            text_embeddings = self._generate_text_embeddings(content_corpus)
            
            # Step 2: Incorporate engagement patterns
            engagement_vector = self._generate_engagement_vector(engagement_patterns)
            
            # Step 3: Apply STAP projection
            semantic_coordinate = self._apply_stap_projection(
                text_embeddings, 
                engagement_vector, 
                user_id
            )
            
            # Step 4: Calculate confidence
            confidence = self._calculate_confidence(text_embeddings, engagement_patterns)
            
            coordinate = SemanticCoordinate(
                user_id=user_id,
                coordinate=semantic_coordinate,
                confidence=confidence,
                last_updated=datetime.now(),
                semantic_version="1.0"
            )
            
            # Cache the coordinate
            self.user_embeddings_cache[user_id] = coordinate
            
            return coordinate
            
        except Exception as e:
            logger.error(f"Error generating semantic coordinate for user {user_id}: {e}")
            # Return default coordinate
            return self._get_default_coordinate(user_id)
    
    def update_semantic_coordinate(self, 
                                 user_id: str,
                                 new_content: List[str],
                                 new_engagements: Dict[str, Any]) -> SemanticCoordinate:
        """
        Update existing semantic coordinate with new data (incremental update).
        """
        try:
            # Get current coordinate
            current_coordinate = self.user_embeddings_cache.get(user_id)
            if not current_coordinate:
                # Fall back to full regeneration
                return self.generate_semantic_coordinate(user_id, new_content, new_engagements)
            
            # Generate embeddings for new content
            new_embeddings = self._generate_text_embeddings(new_content)
            
            if len(new_embeddings) == 0:
                # No new meaningful content, return current coordinate with decayed confidence
                current_coordinate.confidence *= 0.95  # Slight decay
                return current_coordinate
            
            # Incremental STAP update (simplified)
            # In practice, this would use online learning for STAP
            new_coordinate = self._incremental_stap_update(
                current_coordinate.coordinate,
                new_embeddings,
                new_engagements
            )
            
            updated_confidence = self._calculate_confidence(
                new_embeddings, 
                new_engagements,
                base_confidence=current_coordinate.confidence
            )
            
            return SemanticCoordinate(
                user_id=user_id,
                coordinate=new_coordinate,
                confidence=updated_confidence,
                last_updated=datetime.now(),
                semantic_version="1.0"
            )
            
        except Exception as e:
            logger.error(f"Error updating semantic coordinate for user {user_id}: {e}")
            return current_coordinate
    
    def _generate_text_embeddings(self, content_corpus: List[str]) -> np.ndarray:
        """Generate embeddings from text corpus."""
        if not content_corpus:
            return np.array([])
        
        # Filter very short texts
        meaningful_texts = [text for text in content_corpus if len(text.split()) >= 3]
        
        if not meaningful_texts:
            return np.array([])
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(meaningful_texts)
        
        return embeddings
    
    def _generate_engagement_vector(self, engagement_patterns: Dict[str, Any]) -> np.ndarray:
        """Convert engagement patterns to a semantic vector."""
        # This is a simplified implementation
        # In practice, this would be more sophisticated
        
        vector = np.zeros(self.embedding_dim)
        
        if not engagement_patterns:
            return vector
        
        # Use engagement intensity as a scaling factor
        intensity = engagement_patterns.get('engagement_intensity', 0.5)
        
        # Create a pseudo-vector based on engagement patterns
        # This would be learned from data in practice
        engagement_features = [
            engagement_patterns.get('total_engagement', 0) / 1000.0,  # Normalized
            engagement_patterns.get('engagement_intensity', 0.5),
            len(engagement_patterns.get('preferred_content_types', [])),
        ]
        
        # Map to embedding space (simplified)
        if len(engagement_features) > 0:
            # Create a simple projection - in practice, this would be learned
            pseudo_embedding = np.random.RandomState(42).normal(
                0, 0.1, self.embedding_dim
            )
            engagement_weight = np.mean(engagement_features)
            vector = pseudo_embedding * engagement_weight
        
        return vector
    
    def _apply_stap_projection(self, 
                             text_embeddings: np.ndarray, 
                             engagement_vector: np.ndarray,
                             user_id: str) -> np.ndarray:
        """
        Apply STAP projection to create final semantic coordinate.
        This is a simplified version - full STAP would be more complex.
        """
        if len(text_embeddings) == 0:
            # No text data, use engagement vector with noise
            if np.any(engagement_vector):
                return engagement_vector
            else:
                return self._get_random_coordinate(user_id)
        
        # Average text embeddings (basic approach)
        text_coordinate = np.mean(text_embeddings, axis=0)
        
        # Combine with engagement vector
        engagement_weight = 0.2  # How much engagement influences position
        combined = (1 - engagement_weight) * text_coordinate + engagement_weight * engagement_vector
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined
    
    def _incremental_stap_update(self,
                               current_coordinate: np.ndarray,
                               new_embeddings: np.ndarray,
                               new_engagements: Dict[str, Any]) -> np.ndarray:
        """Incremental update of semantic coordinate."""
        # Simplified incremental update
        # In practice, this would use proper online learning
        
        if len(new_embeddings) == 0:
            return current_coordinate
        
        # Average of new embeddings
        new_text_vector = np.mean(new_embeddings, axis=0)
        
        # Moving average update
        learning_rate = 0.1
        updated_coordinate = (1 - learning_rate) * current_coordinate + learning_rate * new_text_vector
        
        # Normalize
        norm = np.linalg.norm(updated_coordinate)
        if norm > 0:
            updated_coordinate = updated_coordinate / norm
        
        return updated_coordinate
    
    def _calculate_confidence(self, 
                            text_embeddings: np.ndarray, 
                            engagement_patterns: Dict[str, Any],
                            base_confidence: float = 0.5) -> float:
        """Calculate confidence in semantic coordinate."""
        confidence = base_confidence
        
        # Text data quality
        if len(text_embeddings) > 0:
            text_quality = min(1.0, len(text_embeddings) / 50.0)  # More texts = higher confidence
            confidence += 0.3 * text_quality
        
        # Engagement data quality
        total_engagement = engagement_patterns.get('total_engagement', 0)
        engagement_quality = min(1.0, total_engagement / 100.0)  # More engagement = higher confidence
        confidence += 0.2 * engagement_quality
        
        return min(1.0, confidence)  # Cap at 1.0
    
    def _get_default_coordinate(self, user_id: str) -> SemanticCoordinate:
        """Get default coordinate for users with insufficient data."""
        coordinate = self._get_random_coordinate(user_id)
        return SemanticCoordinate(
            user_id=user_id,
            coordinate=coordinate,
            confidence=0.1,  # Low confidence for default
            last_updated=datetime.now(),
            semantic_version="1.0"
        )
    
    def _get_random_coordinate(self, user_id: str) -> np.ndarray:
        """Generate deterministic random coordinate based on user_id."""
        # Use user_id as seed for deterministic "randomness"
        seed = hash(user_id) % (2**32)
        rng = np.random.RandomState(seed)
        coordinate = rng.normal(0, 1, self.embedding_dim)
        
        # Normalize
        norm = np.linalg.norm(coordinate)
        if norm > 0:
            coordinate = coordinate / norm
        
        return coordinate

"""
CLASS: Content Recommendation Layer
Generates semantic bridging recommendations
"""
# content_recommendation.py

from word2world_core import *
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommendationLayer:
    """
    Generates content recommendations based on semantic coordinates and user interactions.
    """
    
    def __init__(self, database: DatabaseInterface):
        self.database = database
        self.semantic_knn = NearestNeighbors(n_neighbors=50, metric='cosine')
        self.user_coordinates = {}  # user_id -> coordinate
        self.content_embeddings = {}  # content_id -> embedding
        
        # Recommendation parameters
        self.recommendation_params = {
            'max_recommendations': 10,
            'min_bridge_strength': 0.3,
            'max_semantic_distance': 0.7,
            'exploration_rate': 0.1,
            'diversity_penalty': 0.2
        }
    
    def initialize_semantic_space(self, user_coordinates: Dict[str, SemanticCoordinate]):
        """Initialize the semantic space with user coordinates."""
        self.user_coordinates = user_coordinates
        
        if len(user_coordinates) < 2:
            logger.warning("Insufficient users for semantic space initialization")
            return
        
        # Extract coordinates for KNN
        user_ids = []
        coordinates_list = []
        
        for user_id, coord_obj in user_coordinates.items():
            user_ids.append(user_id)
            coordinates_list.append(coord_obj.coordinate)
        
        coordinates_array = np.array(coordinates_list)
        
        # Build KNN index
        self.semantic_knn.fit(coordinates_array)
        self.user_ids_array = np.array(user_ids)
        
        logger.info(f"Initialized semantic space with {len(user_coordinates)} users")
    
    def generate_recommendations(self, 
                               user_id: str,
                               recommendation_type: str = "bridge",
                               max_recommendations: int = 5) -> List[ContentRecommendation]:
        """
        Generate content recommendations for a user.
        
        Args:
            user_id: Target user ID
            recommendation_type: "bridge", "reinforce", or "explore"
            max_recommendations: Maximum number of recommendations to generate
        """
        if user_id not in self.user_coordinates:
            logger.warning(f"User {user_id} not found in semantic space")
            return []
        
        user_coordinate = self.user_coordinates[user_id]
        
        recommendations = []
        
        if recommendation_type == "bridge":
            recommendations = self._generate_bridge_recommendations(
                user_id, user_coordinate, max_recommendations
            )
        elif recommendation_type == "reinforce":
            recommendations = self._generate_reinforcement_recommendations(
                user_id, user_coordinate, max_recommendations
            )
        elif recommendation_type == "explore":
            recommendations = self._generate_exploration_recommendations(
                user_id, user_coordinate, max_recommendations
            )
        
        # Store recommendations
        for recommendation in recommendations:
            self.database.save_recommendation(recommendation)
        
        return recommendations
    
    def _generate_bridge_recommendations(self,
                                       user_id: str,
                                       user_coordinate: SemanticCoordinate,
                                       max_recommendations: int) -> List[ContentRecommendation]:
        """Generate recommendations that bridge semantic gaps."""
        # Find semantically similar but distinct users
        similar_users = self._find_semantic_neighbors(
            user_coordinate.coordinate, 
            exclude_user=user_id,
            max_neighbors=20
        )
        
        recommendations = []
        
        for neighbor_user_id, distance in similar_users:
            # Check if this is a good bridge candidate
            if not self._is_good_bridge_candidate(user_id, neighbor_user_id, distance):
                continue
            
            # Find bridging content from this user
            bridge_content = self._find_bridging_content(user_id, neighbor_user_id)
            
            if bridge_content:
                bridge_strength = self._calculate_bridge_strength(
                    user_coordinate.coordinate,
                    self.user_coordinates[neighbor_user_id].coordinate,
                    distance
                )
                
                explanation = self._generate_bridge_explanation(
                    user_id, neighbor_user_id, bridge_content
                )
                
                recommendation = ContentRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    user_id=user_id,
                    content_id=bridge_content.content_id,
                    bridge_strength=bridge_strength,
                    semantic_distance=distance,
                    explanation=explanation,
                    recommendation_type="bridge",
                    timestamp=datetime.now()
                )
                
                recommendations.append(recommendation)
                
                if len(recommendations) >= max_recommendations:
                    break
        
        return sorted(recommendations, key=lambda x: x.bridge_strength, reverse=True)
    
    def _generate_reinforcement_recommendations(self,
                                              user_id: str,
                                              user_coordinate: SemanticCoordinate,
                                              max_recommendations: int) -> List[ContentRecommendation]:
        """Generate recommendations that reinforce current interests."""
        # Find very similar users (echo chamber)
        similar_users = self._find_semantic_neighbors(
            user_coordinate.coordinate,
            max_distance=0.3,
            exclude_user=user_id,
            max_neighbors=10
        )
        
        recommendations = []
        
        for neighbor_user_id, distance in similar_users:
            # Find content that aligns with user's existing interests
            reinforcing_content = self._find_reinforcing_content(user_id, neighbor_user_id)
            
            if reinforcing_content:
                bridge_strength = 1.0 - distance  # Higher for closer users
                
                explanation = self._generate_reinforcement_explanation(
                    user_id, neighbor_user_id, reinforcing_content
                )
                
                recommendation = ContentRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    user_id=user_id,
                    content_id=reinforcing_content.content_id,
                    bridge_strength=bridge_strength,
                    semantic_distance=distance,
                    explanation=explanation,
                    recommendation_type="reinforce",
                    timestamp=datetime.now()
                )
                
                recommendations.append(recommendation)
                
                if len(recommendations) >= max_recommendations:
                    break
        
        return sorted(recommendations, key=lambda x: x.bridge_strength, reverse=True)
    
    def _generate_exploration_recommendations(self,
                                            user_id: str,
                                            user_coordinate: SemanticCoordinate,
                                            max_recommendations: int) -> List[ContentRecommendation]:
        """Generate exploratory recommendations outside comfort zone."""
        # Find somewhat distant users (exploration)
        similar_users = self._find_semantic_neighbors(
            user_coordinate.coordinate,
            min_distance=0.5,
            max_distance=0.8,
            exclude_user=user_id,
            max_neighbors=15
        )
        
        recommendations = []
        
        for neighbor_user_id, distance in similar_users:
            # Find content that offers new perspectives
            exploratory_content = self._find_exploratory_content(user_id, neighbor_user_id)
            
            if exploratory_content:
                bridge_strength = distance  # Higher for more distant but reachable
                
                explanation = self._generate_exploration_explanation(
                    user_id, neighbor_user_id, exploratory_content
                )
                
                recommendation = ContentRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    user_id=user_id,
                    content_id=exploratory_content.content_id,
                    bridge_strength=bridge_strength,
                    semantic_distance=distance,
                    explanation=explanation,
                    recommendation_type="explore",
                    timestamp=datetime.now()
                )
                
                recommendations.append(recommendation)
                
                if len(recommendations) >= max_recommendations:
                    break
        
        return sorted(recommendations, key=lambda x: x.bridge_strength, reverse=True)
    
    def _find_semantic_neighbors(self,
                               coordinate: np.ndarray,
                               max_distance: float = 1.0,
                               min_distance: float = 0.0,
                               exclude_user: str = None,
                               max_neighbors: int = 20) -> List[Tuple[str, float]]:
        """Find semantic neighbors within distance range."""
        if not hasattr(self.semantic_knn, '_fit_X'):
            return []
        
        # Query KNN
        distances, indices = self.semantic_knn.kneighbors(
            [coordinate], 
            n_neighbors=min(max_neighbors * 2, len(self.user_ids_array))
        )
        
        neighbors = []
        for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
            neighbor_user_id = self.user_ids_array[index]
            
            # Skip excluded user
            if neighbor_user_id == exclude_user:
                continue
            
            # Check distance constraints
            if min_distance <= distance <= max_distance:
                neighbors.append((neighbor_user_id, distance))
            
            if len(neighbors) >= max_neighbors:
                break
        
        return neighbors
    
    def _is_good_bridge_candidate(self, user_id: str, neighbor_id: str, distance: float) -> bool:
        """Determine if a user is a good bridge candidate."""
        # Check distance range for bridging
        if not (0.2 <= distance <= 0.6):
            return False
        
        # Check if users have interacted before (avoid recommending existing connections)
        # This would query interaction history
        
        return True
    
    def _find_bridging_content(self, user_id: str, neighbor_id: str) -> Optional[UserContent]:
        """Find content that could serve as a bridge between users."""
        # In practice, this would:
        # 1. Get content from neighbor that user hasn't seen
        # 2. Filter for content that's semantically between the users
        # 3. Prioritize well-received content
        
        # Mock implementation
        return UserContent(
            content_id=f"bridge_{neighbor_id}_{uuid.uuid4().hex[:8]}",
            user_id=neighbor_id,
            content_type=ContentType.POST,
            text="This is a bridging content example that might interest both users.",
            timestamp=datetime.now()
        )
    
    def _find_reinforcing_content(self, user_id: str, neighbor_id: str) -> Optional[UserContent]:
        """Find content that reinforces user's existing interests."""
        # Mock implementation
        return UserContent(
            content_id=f"reinforce_{neighbor_id}_{uuid.uuid4().hex[:8]}",
            user_id=neighbor_id,
            content_type=ContentType.POST,
            text="This content aligns with your current interests and perspectives.",
            timestamp=datetime.now()
        )
    
    def _find_exploratory_content(self, user_id: str, neighbor_id: str) -> Optional[UserContent]:
        """Find content that offers new perspectives for exploration."""
        # Mock implementation
        return UserContent(
            content_id=f"explore_{neighbor_id}_{uuid.uuid4().hex[:8]}",
            user_id=neighbor_id,
            content_type=ContentType.POST,
            text="This offers a different perspective that might expand your worldview.",
            timestamp=datetime.now()
        )
    
    def _calculate_bridge_strength(self, coord1: np.ndarray, coord2: np.ndarray, distance: float) -> float:
        """Calculate how strong a bridge could be between coordinates."""
        # Consider distance, coordinate stability, and other factors
        ideal_distance = 0.4  # Optimal bridging distance
        distance_score = 1.0 - abs(distance - ideal_distance) / ideal_distance
        
        return max(0.0, min(1.0, distance_score))
    
    def _generate_bridge_explanation(self, user_id: str, neighbor_id: str, content: UserContent) -> str:
        """Generate human-readable explanation for bridge recommendation."""
        return f"This content from user {neighbor_id} bridges your perspective with adjacent viewpoints, offering common ground while introducing new ideas."
    
    def _generate_reinforcement_explanation(self, user_id: str, neighbor_id: str, content: UserContent) -> str:
        """Generate explanation for reinforcement recommendation."""
        return f"This aligns with your established interests and reinforces your current perspective, providing depth to views you already value."
    
    def _generate_exploration_explanation(self, user_id: str, neighbor_id: str, content: UserContent) -> str:
        """Generate explanation for exploration recommendation."""
        return f"This offers a perspective outside your usual interests, providing an opportunity to explore new ideas and expand your worldview."

"""
CLASS: Feedback Layer 
Optimizes recommendations based on user feedback
"""
# feedback_layer.py

from word2world_core import *
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import statistics

class FeedbackLayer:
    """
    Collects and processes user feedback to continuously improve recommendations.
    """
    
    def __init__(self, database: DatabaseInterface):
        self.database = database
        self.feedback_history = defaultdict(list)
        self.performance_metrics = {
            'click_through_rate': 0.0,
            'engagement_rate': 0.0,
            'satisfaction_score': 0.0,
            'diversity_score': 0.0
        }
        
        # Learning parameters
        self.learning_params = {
            'feedback_decay_rate': 0.95,  # How quickly old feedback decays
            'min_feedback_samples': 10,
            'exploration_bonus': 0.1
        }
    
    def record_feedback(self, 
                       user_id: str, 
                       recommendation_id: str, 
                       feedback_type: str,
                       intensity: float = 1.0,
                       metadata: Dict[str, Any] = None):
        """
        Record user feedback on recommendations.
        
        Args:
            user_id: User providing feedback
            recommendation_id: Which recommendation this feedback is for
            feedback_type: 'click', 'like', 'share', 'save', 'dismiss', 'report'
            intensity: Strength of feedback (0.0 to 1.0)
            metadata: Additional feedback context
        """
        feedback = {
            'user_id': user_id,
            'recommendation_id': recommendation_id,
            'feedback_type': feedback_type,
            'intensity': intensity,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        # Store feedback
        self.feedback_history[user_id].append(feedback)
        
        # Update performance metrics
        self._update_performance_metrics(feedback)
        
        # Apply learning
        self._apply_feedback_learning(user_id, feedback)
        
        logger.info(f"Recorded {feedback_type} feedback from user {user_id}")
    
    def get_recommendation_quality_score(self, recommendation_id: str) -> float:
        """Calculate quality score for a specific recommendation."""
        # Get all feedback for this recommendation
        all_feedback = []
        for user_feedback in self.feedback_history.values():
            for feedback in user_feedback:
                if feedback['recommendation_id'] == recommendation_id:
                    all_feedback.append(feedback)
        
        if not all_feedback:
            return 0.5  # Neutral default
        
        # Calculate weighted score based on feedback types and intensity
        score = 0.0
        total_weight = 0.0
        
        for feedback in all_feedback:
            feedback_score = self._feedback_type_to_score(feedback['feedback_type'])
            weight = feedback['intensity']
            
            score += feedback_score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_score = score / total_weight
        else:
            final_score = 0.5
        
        return max(0.0, min(1.0, final_score))
    
    def get_user_preference_profile(self, user_id: str) -> Dict[str, Any]:
        """Generate preference profile based on feedback history."""
        user_feedback = self.feedback_history.get(user_id, [])
        
        if not user_feedback:
            return {'preferences_learned': False}
        
        # Analyze feedback patterns
        preference_profile = {
            'preferences_learned': True,
            'preferred_recommendation_types': self._analyze_preferred_types(user_feedback),
            'engagement_patterns': self._analyze_engagement_patterns(user_feedback),
            'feedback_consistency': self._calculate_feedback_consistency(user_feedback),
            'exploration_tendency': self._calculate_exploration_tendency(user_feedback),
            'total_feedback_count': len(user_feedback)
        }
        
        return preference_profile
    
    def optimize_recommendation_params(self, 
                                     user_id: str,
                                     current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize recommendation parameters based on user feedback."""
        user_profile = self.get_user_preference_profile(user_id)
        
        if not user_profile['preferences_learned']:
            return current_params  # No optimization without feedback
        
        optimized_params = current_params.copy()
        
        # Adjust based on exploration tendency
        exploration_tendency = user_profile.get('exploration_tendency', 0.5)
        optimized_params['exploration_rate'] = exploration_tendency
        
        # Adjust bridge strength threshold based on engagement
        engagement_level = user_profile['engagement_patterns'].get('average_engagement', 0.5)
        optimized_params['min_bridge_strength'] = max(0.2, 0.5 - engagement_level * 0.3)
        
        # Adjust diversity based on feedback consistency
        consistency = user_profile.get('feedback_consistency', 0.5)
        optimized_params['diversity_penalty'] = 0.2 + (1 - consistency) * 0.3
        
        return optimized_params
    
    def _update_performance_metrics(self, feedback: Dict[str, Any]):
        """Update overall system performance metrics."""
        feedback_type = feedback['feedback_type']
        
        # Update click-through rate (simplified)
        if feedback_type == 'click':
            self.performance_metrics['click_through_rate'] = (
                0.95 * self.performance_metrics['click_through_rate'] + 0.05
            )
        elif feedback_type in ['dismiss', 'report']:
            self.performance_metrics['click_through_rate'] = (
                0.95 * self.performance_metrics['click_through_rate']
            )
        
        # Update engagement rate
        if feedback_type in ['like', 'share', 'save']:
            self.performance_metrics['engagement_rate'] = (
                0.95 * self.performance_metrics['engagement_rate'] + 0.05 * feedback['intensity']
            )
        
        # Update satisfaction score (composite)
        satisfaction_signals = {
            'like': 1.0, 'share': 1.0, 'save': 0.8, 'click': 0.6,
            'dismiss': -0.3, 'report': -1.0
        }
        
        if feedback_type in satisfaction_signals:
            signal = satisfaction_signals[feedback_type] * feedback['intensity']
            self.performance_metrics['satisfaction_score'] = (
                0.9 * self.performance_metrics['satisfaction_score'] + 0.1 * signal
            )
    
    def _apply_feedback_learning(self, user_id: str, feedback: Dict[str, Any]):
        """Apply reinforcement learning based on feedback."""
        # This would update user embeddings or recommendation parameters
        # For now, we'll just log the learning opportunity
        
        recommendation_id = feedback['recommendation_id']
        feedback_type = feedback['feedback_type']
        
        # Positive feedback: reinforce similar recommendations
        if feedback_type in ['like', 'share', 'save']:
            logger.info(f"Reinforcing patterns for user {user_id} based on positive feedback")
        
        # Negative feedback: avoid similar recommendations
        elif feedback_type in ['dismiss', 'report']:
            logger.info(f"Learning to avoid similar patterns for user {user_id}")
    
    def _feedback_type_to_score(self, feedback_type: str) -> float:
        """Convert feedback type to numerical score."""
        scoring = {
            'like': 1.0,
            'share': 1.0,
            'save': 0.9,
            'click': 0.7,
            'view': 0.5,
            'dismiss': -0.5,
            'report': -1.0
        }
        return scoring.get(feedback_type, 0.0)
    
    def _analyze_preferred_types(self, user_feedback: List[Dict]) -> Dict[str, float]:
        """Analyze which recommendation types user prefers."""
        type_scores = defaultdict(list)
        
        for feedback in user_feedback:
            # Get recommendation type from database
            recommendation = self.database.get_recommendation_feedback(
                feedback['recommendation_id']
            )
            if recommendation and 'recommendation_type' in recommendation:
                rec_type = recommendation['recommendation_type']
                score = self._feedback_type_to_score(feedback['feedback_type'])
                type_scores[rec_type].append(score)
        
        # Calculate average scores
        preferred_types = {}
        for rec_type, scores in type_scores.items():
            if scores:
                preferred_types[rec_type] = statistics.mean(scores)
        
        return preferred_types
    
    def _analyze_engagement_patterns(self, user_feedback: List[Dict]) -> Dict[str, float]:
        """Analyze user engagement patterns and intensity."""
        engagement_intensities = [
            feedback['intensity'] 
            for feedback in user_feedback 
            if feedback['feedback_type'] in ['like', 'share', 'save']
        ]
        
        return {
            'average_engagement': statistics.mean(engagement_intensities) if engagement_intensities else 0.5,
            'engagement_consistency': statistics.stdev(engagement_intensities) if len(engagement_intensities) > 1 else 0.0,
            'positive_feedback_ratio': len([
                f for f in user_feedback 
                if self._feedback_type_to_score(f['feedback_type']) > 0
            ]) / max(1, len(user_feedback))
        }
    
    def _calculate_feedback_consistency(self, user_feedback: List[Dict]) -> float:
        """Calculate how consistent user feedback is."""
        if len(user_feedback) < 2:
            return 0.5  # Neutral
        
        scores = [self._feedback_type_to_score(f['feedback_type']) for f in user_feedback]
        
        # Lower standard deviation = more consistent
        stdev = statistics.stdev(scores)
        consistency = 1.0 / (1.0 + stdev)  # Convert to 0-1 scale
        
        return consistency
    
    def _calculate_exploration_tendency(self, user_feedback: List[Dict]) -> float:
        """Calculate user's tendency to explore vs. reinforce."""
        exploration_feedback = [
            f for f in user_feedback 
            if f.get('metadata', {}).get('recommendation_type') == 'explore'
        ]
        
        reinforcement_feedback = [
            f for f in user_feedback 
            if f.get('metadata', {}).get('recommendation_type') == 'reinforce'
        ]
        
        total_relevant = len(exploration_feedback) + len(reinforcement_feedback)
        
        if total_relevant == 0:
            return 0.5  # Balanced default
        
        # Calculate exploration ratio
        exploration_ratio = len(exploration_feedback) / total_relevant
        
        # Weight by engagement
        exploration_engagement = sum(f['intensity'] for f in exploration_feedback)
        reinforcement_engagement = sum(f['intensity'] for f in reinforcement_feedback)
        
        if exploration_engagement + reinforcement_engagement > 0:
            engagement_weighted = exploration_engagement / (exploration_engagement + reinforcement_engagement)
            # Blend simple ratio with engagement-weighted ratio
            return 0.7 * exploration_ratio + 0.3 * engagement_weighted
        
        return exploration_ratio
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'performance_metrics': self.performance_metrics,
            'total_feedback_count': sum(len(feedbacks) for feedbacks in self.feedback_history.values()),
            'active_users_with_feedback': len(self.feedback_history),
            'average_feedback_per_user': (
                sum(len(feedbacks) for feedbacks in self.feedback_history.values()) / 
                max(1, len(self.feedback_history))
            ),
            'feedback_trend': self._calculate_feedback_trend(),
            'recommendation_quality_distribution': self._calculate_quality_distribution()
        }
    
    def _calculate_feedback_trend(self) -> str:
        """Calculate whether feedback is improving over time."""
        # Simplified implementation
        return "stable"  # Would analyze temporal patterns
    
    def _calculate_quality_distribution(self) -> Dict[str, float]:
        """Calculate distribution of recommendation quality scores."""
        # This would analyze all recommendations with feedback
        return {
            'excellent': 0.2,
            'good': 0.5,
            'fair': 0.2,
            'poor': 0.1
        }

"""
Main Word2World Engine - Orchestrates all components
"""
# word2world_engine.py

from word2world_core import *
from user_data_repository import UserDataRepository
from stap_preprocessing import STAPPreprocessingLayer
from content_recommendation import ContentRecommendationLayer
from feedback_layer import FeedbackLayer

class Word2WorldEngine:
    """
    Main orchestrator that integrates all Word2World components.
    """
    
    def __init__(self, database: DatabaseInterface):
        self.database = database
        
        # Initialize components
        self.user_repository = UserDataRepository(database)
        self.stap_processor = STAPPreprocessingLayer()
        self.recommendation_engine = ContentRecommendationLayer(database)
        self.feedback_layer = FeedbackLayer(database)
        
        # Runtime state
        self.user_coordinates = {}
        self.is_initialized = False
        
        logger.info("Word2World Engine initialized")
    
    def initialize_engine(self, user_ids: List[str] = None):
        """Initialize the engine with existing user data."""
        try:
            if user_ids is None:
                # In practice, would fetch active users from database
                user_ids = []
            
            # Load user coordinates
            for user_id in user_ids:
                coordinate = self.database.get_semantic_coordinate(user_id)
                if coordinate:
                    self.user_coordinates[user_id] = coordinate
            
            # Initialize recommendation engine
            self.recommendation_engine.initialize_semantic_space(self.user_coordinates)
            
            self.is_initialized = True
            logger.info(f"Word2World Engine initialized with {len(self.user_coordinates)} users")
            
        except Exception as e:
            logger.error(f"Error initializing Word2World Engine: {e}")
            self.is_initialized = False
    
    def process_user_activity(self, user_id: str, content: UserContent = None, interaction: UserInteraction = None):
        """Process new user activity and update semantic model."""
        try:
            # Store new content
            if content:
                self.user_repository.add_user_content(content)
            
            # Store new interaction
            if interaction:
                self.user_repository.add_user_interaction(interaction)
            
            # Check if we should update semantic coordinate
            should_update = self._should_update_semantic_coordinate(user_id)
            
            if should_update:
                self._update_user_semantic_coordinate(user_id)
                
            logger.info(f"Processed activity for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error processing user activity for {user_id}: {e}")
    
    def generate_recommendations(self, 
                               user_id: str, 
                               recommendation_type: str = "bridge",
                               max_recommendations: int = 5) -> List[ContentRecommendation]:
        """Generate recommendations for a user."""
        if not self.is_initialized:
            self.initialize_engine()
        
        # Get user's preference profile for optimization
        preference_profile = self.feedback_layer.get_user_preference_profile(user_id)
        
        # Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            user_id, recommendation_type, max_recommendations
        )
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations
    
    def record_feedback(self, 
                       user_id: str, 
                       recommendation_id: str, 
                       feedback_type: str,
                       intensity: float = 1.0,
                       metadata: Dict[str, Any] = None):
        """Record user feedback and trigger learning."""
        self.feedback_layer.record_feedback(
            user_id, recommendation_id, feedback_type, intensity, metadata
        )
        
        # Trigger recommendation parameter optimization
        self._optimize_user_recommendations(user_id)
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive insights about a user's semantic position."""
        coordinate = self.user_coordinates.get(user_id)
        if not coordinate:
            return {'error': 'User not found in semantic space'}
        
        engagement_patterns = self.user_repository.get_user_engagement_patterns(user_id)
        preference_profile = self.feedback_layer.get_user_preference_profile(user_id)
        
        return {
            'semantic_coordinate': coordinate.to_dict(),
            'engagement_patterns': engagement_patterns,
            'preference_profile': preference_profile,
            'recommendation_quality': self._calculate_user_recommendation_quality(user_id),
            'semantic_neighbors': self._find_semantic_neighbors(user_id)
        }
    
    def _should_update_semantic_coordinate(self, user_id: str) -> bool:
        """Determine if a user's semantic coordinate should be updated."""
        current_coordinate = self.user_coordinates.get(user_id)
        
        if not current_coordinate:
            return True  # No existing coordinate
        
        # Check time since last update
        time_since_update = datetime.now() - current_coordinate.last_updated
        if time_since_update > timedelta(hours=24):
            return True
        
        # Check if user has significant new activity
        recent_content = self.user_repository.get_user_semantic_corpus(user_id)
        if len(recent_content) > current_coordinate.metadata.get('content_count', 0) + 10:
            return True
        
        return False
    
    def _update_user_semantic_coordinate(self, user_id: str):
        """Update user's semantic coordinate."""
        # Get user data
        content_corpus = self.user_repository.get_user_semantic_corpus(user_id)
        engagement_patterns = self.user_repository.get_user_engagement_patterns(user_id)
        
        # Generate or update coordinate
        if user_id in self.user_coordinates:
            new_coordinate = self.stap_processor.update_semantic_coordinate(
                user_id, content_corpus, engagement_patterns
            )
        else:
            new_coordinate = self.stap_processor.generate_semantic_coordinate(
                user_id, content_corpus, engagement_patterns
            )
        
        # Store coordinate
        self.user_coordinates[user_id] = new_coordinate
        self.database.save_semantic_coordinate(new_coordinate)
        
        # Update recommendation engine
        self.recommendation_engine.initialize_semantic_space(self.user_coordinates)
        
        logger.info(f"Updated semantic coordinate for user {user_id}")
    
    def _optimize_user_recommendations(self, user_id: str):
        """Optimize recommendation parameters for a user based on feedback."""
        current_params = self.recommendation_engine.recommendation_params.copy()
        optimized_params = self.feedback_layer.optimize_recommendation_params(
            user_id, current_params
        )
        
        # Apply optimized parameters
        self.recommendation_engine.recommendation_params.update(optimized_params)
        
        logger.info(f"Optimized recommendation parameters for user {user_id}")
    
    def _calculate_user_recommendation_quality(self, user_id: str) -> float:
        """Calculate average quality of recommendations for a user."""
        # This would analyze the user's feedback history
        user_feedback = self.feedback_layer.feedback_history.get(user_id, [])
        
        if not user_feedback:
            return 0.5  # Neutral default
        
        quality_scores = [
            self.feedback_layer.get_recommendation_quality_score(fb['recommendation_id'])
            for fb in user_feedback
        ]
        
        return statistics.mean(quality_scores) if quality_scores else 0.5
    
    def _find_semantic_neighbors(self, user_id: str, max_neighbors: int = 5) -> List[Dict]:
        """Find and describe semantic neighbors."""
        coordinate = self.user_coordinates.get(user_id)
        if not coordinate:
            return []
        
        # This would use the recommendation engine's KNN
        # For now, return mock data
        return [
            {
                'user_id': f'neighbor_{i}',
                'semantic_distance': 0.1 + i * 0.1,
                'bridge_potential': 0.8 - i * 0.1,
                'common_interests': ['topic_a', 'topic_b']
            }
            for i in range(max_neighbors)
        ]
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health and performance report."""
        return {
            'engine_status': 'active' if self.is_initialized else 'inactive',
            'user_count': len(self.user_coordinates),
            'system_performance': self.feedback_layer.get_system_performance_report(),
            'component_status': {
                'user_repository': 'active',
                'stap_processor': 'active',
                'recommendation_engine': 'active',
                'feedback_layer': 'active'
            },
            'recommendation_metrics': {
                'total_recommendations_generated': 'N/A',  # Would track this
                'average_recommendation_quality': 0.75,    # Would calculate
                'user_engagement_trend': 'improving'       # Would analyze
            }
        }

"""
Flask API for Word2World Engine
"""
# app.py

from flask import Flask, request, jsonify
from word2world_engine import Word2WorldEngine
from word2world_core import *
from database import PostgreSQLDatabase  # Your database implementation

app = Flask(__name__)

# Initialize engine
database = PostgreSQLDatabase("your_connection_string")
word2world_engine = Word2WorldEngine(database)
word2world_engine.initialize_engine()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify(word2world_engine.get_system_health_report())

@app.route('/api/user/<user_id>/activity', methods=['POST'])
def log_user_activity(user_id):
    data = request.json
    
    content = None
    if 'content' in data:
        content = UserContent(
            content_id=data['content'].get('content_id'),
            user_id=user_id,
            content_type=ContentType(data['content']['content_type']),
            text=data['content']['text'],
            timestamp=datetime.fromisoformat(data['content']['timestamp']),
            metadata=data['content'].get('metadata', {})
        )
    
    interaction = None
    if 'interaction' in data:
        interaction = UserInteraction(
            interaction_id=data['interaction'].get('interaction_id'),
            user_id=user_id,
            content_id=data['interaction']['content_id'],
            interaction_type=InteractionType(data['interaction']['interaction_type']),
            timestamp=datetime.fromisoformat(data['interaction']['timestamp']),
            intensity=data['interaction'].get('intensity', 1.0),
            metadata=data['interaction'].get('metadata', {})
        )
    
    word2world_engine.process_user_activity(user_id, content, interaction)
    
    return jsonify({'status': 'success', 'user_id': user_id})

@app.route('/api/user/<user_id>/recommendations', methods=['GET'])
def get_recommendations(user_id):
    recommendation_type = request.args.get('type', 'bridge')
    max_recommendations = int(request.args.get('limit', 5))
    
    recommendations = word2world_engine.generate_recommendations(
        user_id, recommendation_type, max_recommendations
    )
    
    return jsonify({
        'user_id': user_id,
        'recommendations': [rec.__dict__ for rec in recommendations]
    })

@app.route('/api/user/<user_id>/feedback', methods=['POST'])
def submit_feedback(user_id):
    data = request.json
    
    word2world_engine.record_feedback(
        user_id=user_id,
        recommendation_id=data['recommendation_id'],
        feedback_type=data['feedback_type'],
        intensity=data.get('intensity', 1.0),
        metadata=data.get('metadata', {})
    )
    
    return jsonify({'status': 'success', 'user_id': user_id})

@app.route('/api/user/<user_id>/insights', methods=['GET'])
def get_user_insights(user_id):
    insights = word2world_engine.get_user_insights(user_id)
    return jsonify(insights)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
