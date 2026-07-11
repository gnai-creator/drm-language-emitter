import pytest

from drm_language_emitter.config import DRMConfig


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown DRMConfig field"):
        DRMConfig.from_dict({"vocab_size": 32, "not_a_real_field": True})


def test_config_validates_risk_exponent_range():
    with pytest.raises(ValueError, match="risk_exponent_min"):
        DRMConfig(risk_exponent_min=2.0, risk_exponent_max=1.0)
