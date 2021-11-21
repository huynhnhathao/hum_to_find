import logging
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

import torch
import torchaudio

from hum_to_find.core.constants import *

logger = logging.getLogger()

class Evaluator:
    """
    Evaluator class evaluate the given model on the given dataset, using `l2` distance
    or `adaptive`
    """
    def __init__(self, model,
                cached_path: str,
                annotation_file: str,
                audio_dir: str,
                distance_method: str,
                transformation: str,
                target_sample_rate: int, 
                num_samples: int,
                singing_threshold: float,
                device: str) -> None:
        """
        Args:
            model: torch model to extract embeddings
            cached_path: path to save the embedding vectors of audios
            annotation_file: path to the annotation csv
            audio_dir: dir to the audio files
            distance_method: either `l2` or `apdaptive`
            transformation: either `mel_spectrogram` or `spectrogram`
            target_sample_rate: sameple rate use to resample the wave audio
            num_samples: number of samples for each example
            singing_threshold: threshold to consider someone is singing, use to 
                cut head of the wave file
            device: cpu or cuda
        """
        self.model = model
        self.cached_path = cached_path
        self.annotation = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.distance_method = distance_method
        self.transformation = self._get_transformation(transformation)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.singing_threshold = singing_threshold
        self.device = device

    def _get_transformation(self, transformation: str) -> Any:
        transformer = None
        if transformation == 'mel_spectrogram':
            transformer = torchaudio.transforms.MelSpectrogram(
                            sample_rate = SAMPLE_RATE,
                            n_fft = TRANSFORMER_NFFT,
                            hop_length = TRANSFORMER_HOP_LENGTH,
                            n_mels = N_MELS)
        if transformer is None:
            raise ValueError('Transformer not specified')
        return transformer

    def transform_data(self, ) -> None:
        """Embed all the original songs and hummed audios, then save to the 
        cached_dir"""