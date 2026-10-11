"""Netrai's public research API. No model is loaded on import."""
from .api import PipelineConfig, predict_image

__version__ = '0.2.0.dev0'
__all__ = ['PipelineConfig', 'predict_image']
