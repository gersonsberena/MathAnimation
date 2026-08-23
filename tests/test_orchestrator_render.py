import pytest

from orchestrator.render import _parse_resolution


def test_parse_resolution_standard():
    assert _parse_resolution("1080x1920") == (1080, 1920)


def test_parse_resolution_case_insensitive():
    assert _parse_resolution("1080X1920") == (1080, 1920)


def test_parse_resolution_scaled_down_still_9_16():
    assert _parse_resolution("270x480") == (270, 480)


@pytest.mark.parametrize("bad", ["1080", "1080x", "x1920", "abcxdef", "1080-1920"])
def test_parse_resolution_malformed_raises(bad):
    with pytest.raises(ValueError, match="resolution"):
        _parse_resolution(bad)


@pytest.mark.parametrize("bad", ["0x1920", "1080x0", "-1080x1920"])
def test_parse_resolution_non_positive_raises(bad):
    with pytest.raises(ValueError):
        _parse_resolution(bad)


def test_parse_resolution_wrong_aspect_ratio_raises():
    with pytest.raises(ValueError, match="9:16"):
        _parse_resolution("1000x1000")
