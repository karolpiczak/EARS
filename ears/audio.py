# -*- coding: utf-8 -*-

import collections
import json
import logging
import os
import time
import warnings

import librosa
import numpy as np
import sounddevice as sd

from config import *


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

with open('ears/model_labels.json', 'r') as labels_file:
    labels = json.load(labels_file)


signal = np.zeros((AUDIO_DURATION * SAMPLING_RATE, 1), dtype='float32')
spectrogram = np.zeros((MEL_BANDS, AUDIO_DURATION * SAMPLING_RATE // CHUNK_SIZE), dtype='float32')
audio_queue = collections.deque(maxlen=1000)  # Queue for incoming audio blocks
last_chunk = np.zeros((CHUNK_SIZE, 1), dtype='float32')  # Short term memory for the next step

predictions = np.zeros((len(labels), AUDIO_DURATION * SAMPLING_RATE // (BLOCK_SIZE * PREDICTION_STEP)), dtype='float32')
live_audio_feed = collections.deque(maxlen=1)

model = None


def get_raspberry_stats():
    freq = None
    temp = None
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as file:
            temp = int(file.read())
            temp /= 1000.
            temp = np.round(temp, 1)
            temp = '{}\'C'.format(temp)
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as file:
            freq = int(file.read())
            freq /= 1000.
            freq = '{} MHz'.format(int(freq))
    except:
        pass

    return temp, freq


def capture_audio(block, block_len, time, status):
    audio_queue.append(block.copy())


def start():
    # Import classifier model
    logger.info('Initializing a convolutional neural network model...')
    global model

    THEANO_FLAGS = ('device=cpu,'
                    'floatX=float32,'
                    'dnn.conv.algo_bwd_filter=deterministic,'
                    'dnn.conv.algo_bwd_data=deterministic')

    os.environ['THEANO_FLAGS'] = THEANO_FLAGS
    os.environ['KERAS_BACKEND'] = 'theano'

    import keras
    keras.backend.set_image_dim_ordering('th')

    with open('ears/model.json', 'r') as file:
        cfg = file.read()
        model = keras.models.model_from_json(cfg)

    model.load_weights('ears/model.h5')
    logger.debug('Loaded Keras model with weights.')

    # Start audio capture
    sd.default.device = AUDIO_DEVICE
    logger.info('Priming recording device {}.'.format(AUDIO_DEVICE))

    stream = sd.InputStream(channels=1, dtype='float32', callback=capture_audio,
                            samplerate=SAMPLING_RATE, blocksize=BLOCK_SIZE)
    stream.start()

    blocks = []
    processing_queue = collections.deque()

    # Process incoming audio blocks
    while True:
        while len(audio_queue) > 0 and len(blocks) < PREDICTION_STEP:
            blocks.append(audio_queue.popleft())

        if len(blocks) == PREDICTION_STEP:
            new_audio = np.concatenate(blocks)

            # Populate audio for live streaming
            live_audio_feed.append(new_audio[:, 0].copy())

            blocks = []
            processing_queue.append(new_audio)

        if len(processing_queue) > PROCESSING_DELAY + 1:  # +1 for JavaScript streaming delay
            start_time = time.time()

            # Populate audio signal
            step_audio = processing_queue.pop()
            n_samples = len(step_audio)
            signal[:-n_samples] = signal[n_samples:]
            signal[-n_samples:] = step_audio[:]

            # Populate spectrogram
            new_spec = librosa.feature.melspectrogram(np.concatenate([last_chunk, step_audio])[:, 0],
                                                      SAMPLING_RATE, n_fft=FFT_SIZE,
                                                      hop_length=CHUNK_SIZE, n_mels=MEL_BANDS)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # Ignore log10 zero division
                new_spec = librosa.core.perceptual_weighting(new_spec, MEL_FREQS, amin=1e-5,
                                                             ref_power=1e-5, top_db=None)
            new_spec = np.clip(new_spec, 0, 100)
            n_chunks = np.shape(new_spec)[1]
            spectrogram[:, :-n_chunks] = spectrogram[:, n_chunks:]
            spectrogram[:, -n_chunks:] = new_spec

            # Classify incoming audio
            predictions[:, :-1] = predictions[:, 1:]
            offset = SEGMENT_LENGTH // 2
            pred = classify([
                np.stack([spectrogram[:, -(SEGMENT_LENGTH + offset):-offset]]),
                np.stack([spectrogram[:, -SEGMENT_LENGTH:]]),
            ])
            predictions[:, -1] = pred
            target = labels[np.argmax(pred)]

            # Clean up
            last_chunk[:] = step_audio[-CHUNK_SIZE:]

            end_time = time.time()
            time_spent = int((end_time - start_time) * 1000)
            temp, freq = get_raspberry_stats()
            blocks_in_ms = int(PREDICTION_STEP * BLOCK_SIZE / SAMPLING_RATE * 1000)
            msg = '[{}] {}% = {} ms / {} ms ({} blocks) - temp: {} | freq: {} ==> {}'
            timestamp = time.strftime('%H:%M:%S')
            logger.debug(msg.format(timestamp, np.round(time_spent / blocks_in_ms * 100, 1),
                                    time_spent, blocks_in_ms, PREDICTION_STEP, temp, freq, target))

        time.sleep(0.05)


def classify(segments):
    X = np.stack(segments)
    X -= AUDIO_MEAN
    X /= AUDIO_STD
    pred = model.predict(X)
    pred = np.average(pred, axis=0, weights=np.arange(len(pred)) + 1)

    return pred
