"""
Services Package
Collection of inference services for different classification tasks.

Available Services:
- EmotionInferenceService: Emotion classification (Neutral, Sad, Happy, Angry)
- GenderInferenceService: Speaker gender identification (Male, Female)
- IntentInferenceService: User intent classification (SLURP dataset)
- SpeakerInferenceService: Speaker identification
"""

from .emotion import EmotionInferenceService
from .gender import GenderInferenceService
from .intent import IntentInferenceService
from .speaker import SpeakerInferenceService

__all__ = [
    "EmotionInferenceService",
    "GenderInferenceService",
    "IntentInferenceService",
    "SpeakerInferenceService",
]
