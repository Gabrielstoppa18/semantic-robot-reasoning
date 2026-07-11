import pytest

from reasoning.vlm_pick_and_place import _cube_name_for_label


def test_cube_name_for_label_matches_blue():
    assert _cube_name_for_label("the blue cube") == "water_cube"


def test_cube_name_for_label_matches_red():
    assert _cube_name_for_label("a red cube on the left") == "fire_cube"


def test_cube_name_for_label_is_case_insensitive():
    assert _cube_name_for_label("BLUE CUBE") == "water_cube"


def test_cube_name_for_label_raises_for_unknown_color():
    with pytest.raises(ValueError):
        _cube_name_for_label("a green sphere")
