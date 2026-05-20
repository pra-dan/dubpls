from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union

class AudioClassification(BaseModel):
    label: str
    confidence: Optional[float] = None

class Segment(BaseModel):
    text: str = Field(description="The original English speech text")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    speaker: str = Field(description="Speaker identifier")
    collected_scenes_path: Optional[str] = Field(None, description="Path to extracted video clip")
    
    # Allow dict or string for backwards compatibility with existing JSON
    audio_gender_classification: Optional[Union[AudioClassification, str, dict]] = None
    audio_emotion_classification: Optional[Union[AudioClassification, str, dict]] = None
    
    video_context: Optional[str] = Field(None, description="Visual context description extracted by VLM")
    
    translation: Optional[str] = Field(None, description="The translated text")
    fr: Optional[str] = Field(None, description="Alternative key for French translation (legacy)")
    fr_gt: Optional[str] = Field(None, description="Ground truth French translation for evaluation")

class PipelineData(BaseModel):
    segments: List[Segment]
