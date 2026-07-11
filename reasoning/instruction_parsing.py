"""Extracts a target cube color from a free-text instruction.

This is deliberately a simple keyword lookup, not an LLM call -- the VLM is
reserved for *visual* grounding (see `scene_recognition.py`); picking out
"the user wants the {color} cube" from their own typed instruction doesn't
need a model at all yet. Bilingual (English/Portuguese) since the dashboard
this feeds is used in Portuguese.
"""

# Ordered so a color word that's a substring of another (there are none here,
# but keep this in mind if more colors/keywords are added) doesn't shadow it.
_COLOR_KEYWORDS = {
    "blue": "blue",
    "azul": "blue",
    "red": "red",
    "vermelho": "red",
    "vermelha": "red",
}


def extract_target_color(instruction):
    """Return "blue"/"red" if a matching color keyword appears in
    `instruction` (case-insensitive), else None."""
    text = instruction.lower()
    for keyword, color in _COLOR_KEYWORDS.items():
        if keyword in text:
            return color
    return None
