"""Unit tests on synthetic rows; no real data."""
import pytest

from bridgedx import cohort


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_icd10_3char():
    assert cohort.icd10_3char("I10.9") == "I10"
    assert cohort.icd10_3char("E11.65") == "E11"
    assert cohort.icd10_3char("401.9") is None
