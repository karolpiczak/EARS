# -*- coding: utf-8 -*-

AUDIO_DEVICE = 'H1'  # Recording device name as listed by `python -m sounddevice`
RECORDING_PATH = '/storage'  # Recording storage path

SAMPLING_RATE = 44100  # Audio sampling rate, other parameters are hand-tuned for 44.1 kHz
BLOCK_SIZE = 8192  # Size of sound device audio capture buffer
