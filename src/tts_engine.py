# src/tts_engine.py
# Urdu Text-to-Speech using gTTS + pygame
# Plays Urdu audio with cooldown to avoid repeated speech

from gtts import gTTS
import pygame
import tempfile
import os
import time

pygame.mixer.init()

_last_spoken = ""
_last_time = 0
COOLDOWN = 2.0  # seconds between utterances


def speak_urdu(urdu_text: str):
    """
    Speaks the given Urdu script text out loud.
    Skips if same text was spoken recently (within cooldown).
    """
    global _last_spoken, _last_time

    if not urdu_text or not urdu_text.strip():
        return

    now = time.time()
    if urdu_text == _last_spoken and (now - _last_time) < COOLDOWN:
        return

    _last_spoken = urdu_text
    _last_time = now

    try:
        tts = gTTS(text=urdu_text, lang="ur", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
            tts.save(tmp_path)

        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()

        # Wait for audio to finish before deleting temp file
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        os.remove(tmp_path)

    except Exception as e:
        print(f"[TTS error] {e}")


def set_cooldown(seconds: float):
    """Adjust cooldown between utterances."""
    global COOLDOWN
    COOLDOWN = seconds
