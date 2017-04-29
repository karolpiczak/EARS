# -*- coding: utf-8 -*-

import librosa


AUDIO_DEVICE = 'H1'  # Recording device name as listed by `python -m sounddevice`

AUDIO_DURATION = 10  # Duration of audio material to retain, in seconds

SAMPLING_RATE = 44100  # Audio sampling rate, other parameters are hand-tuned for 44.1 kHz
CHUNK_SIZE = 882  # Spectrogram hop_size, 882 samples @ 44.1 kHz = 20 ms
FFT_SIZE = 2 * CHUNK_SIZE  # Spectrogram FFT window length
BLOCK_SIZE = 8 * CHUNK_SIZE  # Size of sound device audio capture buffer
PREDICTION_STEP = 6  # How often new predictions should be output, in blocks
PREDICTION_STEP_IN_MS = int(PREDICTION_STEP * BLOCK_SIZE / SAMPLING_RATE * 1000)
SEGMENT_LENGTH = 100  # Lookback window for classification, in chunks, 100 @ 20 ms = 2 s

PROCESSING_DELAY = 3  # Audio streaming delay compensation, in processing steps

MEL_BANDS = 80  # Number of mel frequency bands
MEL_FREQS = librosa.core.mel_frequencies(n_mels=MEL_BANDS)

AUDIO_MEAN = 20.0
AUDIO_STD = 20.0
