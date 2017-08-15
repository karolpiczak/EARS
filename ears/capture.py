# -*- coding: utf-8 -*-

import collections
import datetime
import logging
import os
import sched
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import *


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


audio_queue = collections.deque()  # Queue for incoming audio blocks


def capture_audio(block, block_len, time, status):
    audio_queue.append(block.copy())


def save_audio():
    global t0

    folder = t0.strftime('%Y%m%d')
    begin = t0.strftime('%Y%m%d-%H%M%S')
    t0 = datetime.datetime.now()
    end = t0.strftime('%Y%m%d-%H%M%S')

    logger.info(f'Saving recording {begin}.')

    n_blocks = len(audio_queue)
    signal = np.zeros((n_blocks * BLOCK_SIZE, 1), dtype='float32')

    for i in range(n_blocks):
        block = audio_queue.popleft()
        left = i * BLOCK_SIZE
        right = (i + 1) * BLOCK_SIZE
        signal[left:right, 0] = block[:, 0]

    if not os.path.exists(f'{RECORDING_PATH}/{folder}'):
        os.mkdir(f'{RECORDING_PATH}/{folder}')

    sf.write(f'{RECORDING_PATH}/{folder}/{begin}.flac', signal, SAMPLING_RATE,
             format='flac', subtype='PCM_24')

    logger.info(f'Starting new recording {end}.')
    scheduler.enterabs(get_next_time(t0).timestamp(), 1, save_audio)


def get_next_time(now):
    t1 = datetime.datetime(now.year, now.month, now.day, now.hour, now.minute - now.minute % 10, 0)
    return t1 + datetime.timedelta(minutes=10)


if __name__ == '__main__':
    sd.default.device = AUDIO_DEVICE
    logger.info(f'Priming recording device {AUDIO_DEVICE}')

    stream = sd.InputStream(channels=1, dtype='float32', callback=capture_audio,
                            samplerate=SAMPLING_RATE, blocksize=BLOCK_SIZE)
    stream.start()

    t0 = datetime.datetime.now()
    logger.info(f'[{t0}] Recording started')

    scheduler = sched.scheduler(timefunc=time.time)

    scheduler.enterabs(get_next_time(t0).timestamp(), 1, save_audio)
    scheduler.run()
