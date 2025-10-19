"""
Speaker identification and tracking
"""

import hashlib
import logging
import time
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class SpeakerIdentifier:
    """Identify and track speakers using voice embeddings"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        ttl_hours: float = 1.0
    ):
        """
        Initialize speaker identifier
        
        Args:
            similarity_threshold: Cosine similarity threshold for matching (0.0-1.0)
            ttl_hours: Time-to-live for speaker embeddings in hours
        """
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_hours * 3600
        
        # Storage for known speakers
        self.speakers: Dict[str, Dict] = {}
        # Format: {speaker_id: {"embedding": np.ndarray, "last_seen": float, "count": int}}
        
        logger.info(f"SpeakerIdentifier initialized "
                   f"(threshold={similarity_threshold}, ttl={ttl_hours}h)")
    
    def identify(self, embedding: np.ndarray) -> str:
        """
        Identify speaker from embedding, or create new speaker ID
        
        Args:
            embedding: Speaker embedding vector
            
        Returns:
            Speaker ID (6-character hash)
        """
        # Clean up expired speakers
        self._cleanup_expired()
        
        # If no known speakers, create new one
        if not self.speakers:
            speaker_id = self._create_new_speaker(embedding)
            logger.debug(f"New speaker registered: {speaker_id}")
            return speaker_id
        
        # Compare with known speakers
        best_match_id = None
        best_similarity = -1.0
        
        for speaker_id, speaker_data in self.speakers.items():
            known_embedding = speaker_data["embedding"]
            similarity = self._compute_similarity(embedding, known_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = speaker_id
        
        # Check if best match exceeds threshold
        if best_similarity >= self.similarity_threshold:
            # Match found - update speaker
            self.speakers[best_match_id]["last_seen"] = time.time()
            self.speakers[best_match_id]["count"] += 1
            logger.debug(f"Speaker matched: {best_match_id} "
                        f"(similarity={best_similarity:.3f})")
            return best_match_id
        else:
            # No match - create new speaker
            speaker_id = self._create_new_speaker(embedding)
            logger.debug(f"New speaker registered: {speaker_id} "
                       f"(best_match_similarity={best_similarity:.3f})")
            return speaker_id
    
    def _create_new_speaker(self, embedding: np.ndarray) -> str:
        """
        Create new speaker with unique ID
        
        Args:
            embedding: Speaker embedding vector
            
        Returns:
            New speaker ID
        """
        # Generate unique ID based on embedding hash and timestamp
        hash_input = f"{embedding.tobytes()}{time.time()}".encode()
        speaker_id = hashlib.sha256(hash_input).hexdigest()[:6]
        
        # Store speaker data
        self.speakers[speaker_id] = {
            "embedding": embedding,
            "last_seen": time.time(),
            "count": 1
        }
        
        return speaker_id
    
    def _compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity (0.0-1.0)
        """
        similarity = cosine_similarity(embedding1, embedding2)
        return float(similarity)
    
    def _cleanup_expired(self):
        """Remove expired speakers based on TTL"""
        current_time = time.time()
        expired_speakers = []
        
        for speaker_id, speaker_data in self.speakers.items():
            if current_time - speaker_data["last_seen"] > self.ttl_seconds:
                expired_speakers.append(speaker_id)
        
        for speaker_id in expired_speakers:
            logger.debug(f"Speaker {speaker_id} expired (TTL exceeded)")
            del self.speakers[speaker_id]
    
    def get_speaker_info(self, speaker_id: str) -> Optional[Dict]:
        """Get information about a speaker"""
        if speaker_id in self.speakers:
            data = self.speakers[speaker_id]
            return {
                "speaker_id": speaker_id,
                "count": data["count"],
                "last_seen": data["last_seen"],
                "age_seconds": time.time() - data["last_seen"]
            }
        return None
    
    def get_all_speakers(self) -> list:
        """Get information about all known speakers"""
        speakers = []
        for speaker_id in self.speakers:
            info = self.get_speaker_info(speaker_id)
            if info:
                speakers.append(info)
        return speakers
    
    def reset(self):
        """Clear all speaker data"""
        self.speakers.clear()
        logger.debug("All speaker data cleared")
