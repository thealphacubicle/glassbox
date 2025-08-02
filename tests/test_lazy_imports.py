import pytest
from glassbox.utils.lazy_imports import optional_import


def test_optional_import_success():
    math = optional_import("math")
    assert math.sqrt(4) == 2


def test_optional_import_failure():
    with pytest.raises(ImportError):
        optional_import("definitely_missing_module")
