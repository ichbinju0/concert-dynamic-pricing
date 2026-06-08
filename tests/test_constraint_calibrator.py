import pytest
from tools.constraint_calibrator import get_price_bounds


def test_popularity_7():
    result = get_price_bounds(7, 154000)
    assert result["floor"] == 138600
    assert result["ceiling"] == 385000


def test_popularity_6():
    result = get_price_bounds(6, 154000)
    assert result["floor"] == 130900
    assert result["ceiling"] == 308000


def test_popularity_4():
    result = get_price_bounds(4, 154000)
    assert result["floor"] == 123200
    assert result["ceiling"] == 231000


def test_popularity_2():
    result = get_price_bounds(2, 154000)
    assert result["floor"] == 107800
    assert result["ceiling"] == 184800


def test_keys_are_int():
    result = get_price_bounds(6, 100000)
    assert isinstance(result["floor"], int)
    assert isinstance(result["ceiling"], int)
