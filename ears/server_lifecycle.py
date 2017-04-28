# -*- coding: utf-8 -*-

import threading

import audio


def on_server_loaded(server_context):
    detector_thread = threading.Thread(target=audio.start)
    detector_thread.daemon = True
    detector_thread.start()
