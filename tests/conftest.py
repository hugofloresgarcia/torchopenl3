from pathlib import Path
import os

import pytest

import torchopenl3
import audio_utils as au


TEST_ASSETS_DIR = Path(__file__).parent / 'assets'


###############################################################################
# Pytest fixtures
###############################################################################

@pytest.fixture(scope='session')
def example_audio():
    """preload example audio"""
    return au.io.load_audio_file(os.path.join(TEST_ASSETS_DIR, 'example_audio.wav'))