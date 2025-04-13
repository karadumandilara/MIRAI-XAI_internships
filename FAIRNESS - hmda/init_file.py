
"""
HMDA Fairness Analysis Package

This package provides tools for analyzing fairness in the Home Mortgage Disclosure Act (HMDA) dataset,
focusing on gender-based fairness in mortgage lending decisions.

Modules:
    - data_processor: Processing and preparing the HMDA data
    - feature_selection: Feature selection pipeline (optional, assumed external)
    - fairness_models: Implementing fair and unfair classification models
    - fairness_metrics: Calculating AIF360-based fairness and performance metrics
    - visualization: Visualizations for model comparison and fairness dashboards
    - run_analysis: Script to run the entire pipeline end-to-end
"""

from .data_processor import HMDADataProcessor
from .fairness_models import FairnessModels
from .fairness_metrics import FairnessMetrics
from .visualization import FairnessVisualizer

__all__ = [
    "HMDADataProcessor",
    "FairnessModels",
    "FairnessMetrics",
    "FairnessVisualizer"
]