"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .cityscapes13 import City13Segmentation, City13SegmentationForMorph
from .densepass13 import DensePASS13Segmentation, DensePASS13SegmentationForMorph
from .stanford2d3d8 import Stanford2d3d8Segmentation
from .stanford2d3d_pan8 import Stanford2d3dPan8Segmentation
from .structured3d8 import Structured3d8Segmentation
from .synpass13 import SynPASS13Segmentation, SynPASS13SegmentationForMorph

datasets = {
    'cityscape13': City13Segmentation,
    'synpass13': SynPASS13Segmentation,
    'densepass13': DensePASS13Segmentation,
    'stanford2d3d8': Stanford2d3d8Segmentation,
    'structured3d8': Structured3d8Segmentation,
    'stanford2d3d_pan8': Stanford2d3dPan8Segmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
