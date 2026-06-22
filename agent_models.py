from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import ConfigDict


class IdiomFlag(BaseModel):
    """
    Represents a single idiom, slang phrase, or culturally-specific expression
    detected in a dialogue line. Produced by the VLM context agent and consumed
    by the translation agent.
    """
    phrase: str = Field(
        description="The exact English phrase that is idiomatic/slang (e.g. 'opsy-daisy', 'tick-tick', 'Whiskey Dick')"
    )
    meaning: str = Field(
        description="Plain-English literal meaning of the phrase (e.g. 'here we go / I am about to grab you now')"
    )
    should_substitute: bool = Field(
        description="True if the phrase MUST be culturally substituted (never transliterated). False if transliteration is acceptable."
    )


class VideoProfile(BaseModel):
    """Whole-video metadata extracted by the VLM once, reused per segment."""
    genre: str = Field(
        description="Genre of the content, e.g. 'comedic action', 'historical drama', 'sci-fi thriller'"
    )
    genre_examples: str = Field(
        description="Comma-separated genre tag examples for translator context, e.g. 'Comedy, Action, Buddy Cop'"
    )
    video_type: Literal["movie", "trailer", "speech", "tutorial", "documentary", "series", "short_film", "other"] = Field(
        description="Classification of what kind of video this is"
    )
    maturity_rating: str = Field(
        description="Content maturity level, e.g. 'R-rated', 'PG-13', 'G', 'NC-17'"
    )
    tone: str = Field(
        description="Overall emotional/stylistic tone, e.g. 'sarcastic', 'serious', 'urgent', 'comedic', 'dark'"
    )
    formality_level: str = Field(
        description="Language formality, e.g. 'informal', 'formal', 'neutral', 'street slang'"
    )


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
    
    video_context: Optional[str] = Field(None, description="Visual context description extracted by VLM")
    dialogue_justification: Optional[str] = Field(None, description="VLM's justification for why the speaker said this English line, based on the mosaic")
    # Whole-video profile shared across all segments (populated from PipelineData.video_profile)
    video_profile: Optional["VideoProfile"] = Field(None, description="Global video genre/type profile from VLM")
    
    # Per-segment character & dubbing style are removed to prevent overriding video profile.
    idiom_flags: Optional[List[IdiomFlag]] = Field(None, description="Idiom/slang phrases detected in the English line, with plain-English meanings and substitution guidance")
    
    translation: Optional[str] = Field(None, description="The translated text")
    
    # Target language dynamically passed in
    target_language: str = Field("fr", description="Target language code (e.g. fr, hi)")
    language_name: str = Field("French", description="Full name of target language (e.g. French)")

    # Allow dynamic ground truth keys like fr_gt, hi_gt
    model_config = ConfigDict(extra="allow")

class PipelineData(BaseModel):
    segments: List[Segment]
    video_profile: Optional[VideoProfile] = Field(
        None,
        description="Whole-video structural/genre profile extracted by VLM at the start of Stage 1"
    )
