# src/urdu_renderer.py
# Renders RTL Urdu text on OpenCV frames using PIL with raqm layout engine

from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2

FONT_PATH = r"C:\Windows\Fonts\Jameel Noori Nastaleeq.ttf"
FONT_LARGE = ImageFont.truetype(FONT_PATH, 42)
FONT_SMALL = ImageFont.truetype(FONT_PATH, 28)


def render_urdu_on_frame(frame, urdu_text: str, position: tuple = (20, 20),
                          font_size: str = "large",
                          color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    Renders Urdu text on an OpenCV frame using raqm layout engine.
    position: (x, y) from top-left
    font_size: "large" or "small"
    color: RGB tuple
    Returns modified frame.
    """
    if not urdu_text or not urdu_text.strip():
        return frame

    font = FONT_LARGE if font_size == "large" else FONT_SMALL

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Shadow for readability
    draw.text((position[0] + 2, position[1] + 2),
            urdu_text,
            font=font,
            fill=(0, 0, 0))
    # Main text
    draw.text(position,
          urdu_text,
          font=font,
          fill=color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def render_urdu_centered(frame, urdu_text: str, y: int = 50,
                          font_size: str = "large",
                          color: tuple = (255, 255, 255)) -> np.ndarray:
    """Renders Urdu text horizontally centered on frame."""
    if not urdu_text or not urdu_text.strip():
        return frame

    font = FONT_LARGE if font_size == "large" else FONT_SMALL

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    bbox = draw.textbbox((0, 0), urdu_text, font=font, language="ur", direction="rtl")
    text_width = bbox[2] - bbox[0]
    frame_width = frame.shape[1]
    x = max(0, (frame_width - text_width) // 2)

    draw.text((x + 2, y + 2), urdu_text, font=font,
              fill=(0, 0, 0), language="ur", direction="rtl")
    draw.text((x, y), urdu_text, font=font,
              fill=color, language="ur", direction="rtl")

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
