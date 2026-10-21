import numpy as np
import pytest

from tracknet.features import (color_histogram, cosine_similarity, direction_compatibility,
                               direction_from_trajectory, dominant_color_name)
from tests.conftest import patch


@pytest.mark.parametrize("name", ["red", "white", "blue", "black", "silver", "green", "yellow", "orange", "purple"])
def test_colour_names_are_recovered(name):
    assert dominant_color_name(patch(name)) == name


def test_empty_crop_is_unknown():
    assert dominant_color_name(np.zeros((0, 0, 3), dtype=np.uint8)) == "unknown"


def test_histogram_is_unit_norm_and_similarity_orders_correctly():
    red1, red2, blue = (color_histogram(patch(c)) for c in ("red", "red", "blue"))
    assert abs(np.linalg.norm(red1) - 1.0) < 1e-4
    assert cosine_similarity(red1, red2) > 0.99
    assert cosine_similarity(red1, blue) < 0.2


def test_cosine_handles_degenerate_input():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    assert cosine_similarity([1, 0], [1, 0, 0]) == 0.0


def test_direction_from_trajectory():
    assert direction_from_trajectory([[0, 0, 100], [5, 300, 105]]) == "right"
    assert direction_from_trajectory([[0, 300, 100], [5, 0, 95]]) == "left"
    assert direction_from_trajectory([[0, 100, 50], [5, 105, 300]]) == "approaching"
    assert direction_from_trajectory([[0, 100, 300], [5, 100, 40]]) == "receding"
    assert direction_from_trajectory([[0, 100, 100], [5, 104, 101]]) == "static"
    assert direction_from_trajectory([[0, 1, 1]]) == "static"


def test_direction_compatibility():
    assert direction_compatibility("left", "left") == 1.0
    assert direction_compatibility("left", "right") == 0.0
    assert direction_compatibility("static", "right") == 0.5
    assert direction_compatibility("left", "approaching") == 0.2
