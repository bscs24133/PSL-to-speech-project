# src/tts_engine.py
from gtts import gTTS
import pygame
import tempfile
import os
import time

pygame.mixer.init()

_last_spoken = ""
_last_time = 0
COOLDOWN = 2.0

def speak_urdu(urdu_text: str):
    global _last_spoken, _last_time

    if not urdu_text or not urdu_text.strip():
        return

    now = time.time()
    if urdu_text == _last_spoken and (now - _last_time) < COOLDOWN:
        return

    _last_spoken = urdu_text
    _last_time = now
    tmp_path = None

    try:
        tts = gTTS(text=urdu_text, lang="ur", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
            tts.save(tmp_path)
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"[TTS error] {e}")
    finally:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            time.sleep(0.1)
        except Exception:
            pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def set_cooldown(seconds: float):
    global COOLDOWN
    COOLDOWN = seconds