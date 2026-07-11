import pytest

from drm_language_emitter.config import DRMConfig


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown DRMConfig field"):
        DRMConfig.from_dict({"vocab_size": 32, "not_a_real_field": True})


def test_config_validates_risk_exponent_range():
    with pytest.raises(ValueError, match="risk_exponent_min"):
        DRMConfig(risk_exponent_min=2.0, risk_exponent_max=1.0)


def test_config_validates_geometry_update_interval():
    assert DRMConfig.from_dict({"vocab_size": 32}).geometry_update_interval == 1
    with pytest.raises(ValueError, match="geometry_update_interval"):
        DRMConfig(geometry_update_interval=0)


def test_config_validates_factorized_basis_sizes():
    config = DRMConfig(direction_basis_size=4, metric_u_basis_size=5)
    assert config.direction_basis_size == 4
    assert config.metric_u_basis_size == 5
    with pytest.raises(ValueError, match="direction_basis_size"):
        DRMConfig(direction_basis_size=-1)


def test_config_validates_bptt_truncate_interval():
    assert DRMConfig(bptt_truncate_interval=8).bptt_truncate_interval == 8
    with pytest.raises(ValueError, match="bptt_truncate_interval"):
        DRMConfig(bptt_truncate_interval=-1)
