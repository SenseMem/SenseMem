"""Model contracts and optional local embedding and speech adapters."""

from sensemem.models.base import (
    EmbeddingBackend,
    EmbedTask,
    FaceBackend,
    GenerationBackend,
    ModelInput,
    SpeechBackend,
    TranscriptionBackend,
)
from sensemem.models.funasr import (
    DEFAULT_FUNASR_MODEL_ID,
    DEFAULT_FUNASR_MODEL_REVISION,
    DEFAULT_FUNASR_RECIPE,
    DEFAULT_FUNASR_SPEAKER_MODEL_ID,
    DEFAULT_FUNASR_SPEAKER_REVISION,
    DEFAULT_FUNASR_VAD_MODEL_ID,
    DEFAULT_FUNASR_VAD_REVISION,
    FunASRRecipe,
    FunASRTranscriber,
)
from sensemem.models.opencv_face import OpenCVFaceAnalyzer
from sensemem.models.sentence_transformers import SentenceTransformersEmbedder
from sensemem.perception import (
    FaceAnalysis,
    FaceEmbedding,
    SpeakerEmbedding,
    SpeechAnalysis,
    SpeechTurn,
)

__all__ = [
    "DEFAULT_FUNASR_MODEL_ID",
    "DEFAULT_FUNASR_MODEL_REVISION",
    "DEFAULT_FUNASR_RECIPE",
    "DEFAULT_FUNASR_SPEAKER_MODEL_ID",
    "DEFAULT_FUNASR_SPEAKER_REVISION",
    "DEFAULT_FUNASR_VAD_MODEL_ID",
    "DEFAULT_FUNASR_VAD_REVISION",
    "EmbedTask",
    "EmbeddingBackend",
    "FaceAnalysis",
    "FaceBackend",
    "FaceEmbedding",
    "FunASRRecipe",
    "FunASRTranscriber",
    "GenerationBackend",
    "ModelInput",
    "OpenCVFaceAnalyzer",
    "SentenceTransformersEmbedder",
    "SpeakerEmbedding",
    "SpeechAnalysis",
    "SpeechBackend",
    "SpeechTurn",
    "TranscriptionBackend",
]
