"""Thin wrapper around a local Ollama vision-language model (VLM).

Requires Ollama running locally (default install listens on
http://localhost:11434) with a vision-capable model already pulled:

    ollama pull llava

This is the only place that talks to Ollama directly; `scene_recognition.py`
builds prompts/parses responses on top of `ask` and `ask_json`.

`llama3.2-vision` was tried first (per the thesis's original model choice)
but this Ollama build (0.31.2, installed via winget, no newer version
available at the time) fails to load it at all: `ollama run llama3.2-vision`
itself errors with "unknown model architecture: 'mllama'" -- a bug/gap in
this Ollama build's support for that specific model family, not something
fixable from this codebase. `llava` uses a different, more mature code path
in Ollama and loads/runs fine. Re-check whether `llama3.2-vision` works
before assuming this limitation still applies if Ollama is ever upgraded.
"""

import json

import cv2
import ollama

DEFAULT_MODEL = "llava"


def encode_image(image_rgb):
    """Encode an RGB numpy array (as returned by
    `perception.stereo_localization.get_image`) as PNG bytes for Ollama."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("failed to PNG-encode image for the VLM")
    return buffer.tobytes()


def ask(image_rgb, prompt, model=DEFAULT_MODEL):
    """Send one image + text prompt to the VLM, return its text response."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [encode_image(image_rgb)]}],
    )
    return response["message"]["content"]


def ask_json(image_rgb, prompt, schema, model=DEFAULT_MODEL):
    """Like `ask`, but constrains the response to the given JSON schema
    (Ollama's structured-output support) and returns it already parsed."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [encode_image(image_rgb)]}],
        format=schema,
    )
    return json.loads(response["message"]["content"])
