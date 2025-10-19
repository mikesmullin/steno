"""
Speaker embedding extraction using SpeechBrain ECAPA-TDNN
"""

import logging
from typing import Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SpeakerEmbeddings:
    """Extract speaker embeddings from audio"""
    
    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "auto"
    ):
        """
        Initialize speaker embedding model
        
        Args:
            model_source: SpeechBrain model source or local path
            device: Device to run on (auto/cpu/cuda/mps)
        """
        # Auto-detect device if needed
        if device == "auto":
            device = self._detect_device()
        
        self.device = device
        
        logger.info(f"Initializing speaker embedding model (device={device})")
        
        try:
            # Set environment variable to avoid symlink issues on Windows
            import os
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            
            # Use copy strategy instead of symlink for Windows compatibility
            from speechbrain.utils.fetching import LocalStrategy
            from speechbrain.inference import EncoderClassifier
            
            # Load SpeechBrain ECAPA-TDNN model
            self.model = EncoderClassifier.from_hparams(
                source=model_source,
                savedir="tmp/speaker_models",
                run_opts={"device": device},
                local_strategy=LocalStrategy.COPY  # Use copy instead of symlink
            )
            logger.info("Speaker embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load speaker embedding model: {e}")
            raise
    
    def _detect_device(self) -> str:
        """Auto-detect best available device"""
        try:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except:
            return "cpu"
    
    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio
        
        Args:
            audio: Audio array (float32, normalized)
            sample_rate: Audio sample rate
            
        Returns:
            Speaker embedding vector (192-dim for ECAPA), or None on error
        """
        try:
            # Convert numpy to torch tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Ensure correct shape (add batch dimension if needed)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_tensor)
            
            # Convert to numpy and flatten
            embedding_np = embedding.squeeze().cpu().numpy()
            
            logger.debug(f"Extracted embedding (shape: {embedding_np.shape})")
            
            return embedding_np
            
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            return None
