from types import SimpleNamespace

import pytest

from neuralop.training import torch_setup


class LegacyTorch:
    def __init__(self):
        self.precision = None
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=None)),
            cudnn=SimpleNamespace(allow_tf32=None),
        )

    def set_float32_matmul_precision(self, precision):
        self.precision = precision


class NewTorch:
    def __init__(self):
        self.backends = SimpleNamespace(
            fp32_precision="none",
            cuda=SimpleNamespace(matmul=SimpleNamespace(fp32_precision="tf32")),
            cudnn=SimpleNamespace(
                fp32_precision="none",
                conv=SimpleNamespace(fp32_precision="tf32"),
                rnn=SimpleNamespace(fp32_precision="tf32"),
            ),
        )


@pytest.mark.parametrize(
    ("allow_tf32", "expected_precision"),
    [(False, "highest"), (True, "high")],
)
def test_set_tf32_legacy_api(monkeypatch, allow_tf32, expected_precision):
    fake_torch = LegacyTorch()
    monkeypatch.setattr(torch_setup, "torch", fake_torch)

    torch_setup.set_tf32(allow_tf32)

    assert fake_torch.precision == expected_precision
    assert fake_torch.backends.cuda.matmul.allow_tf32 is allow_tf32
    assert fake_torch.backends.cudnn.allow_tf32 is allow_tf32


@pytest.mark.parametrize(
    ("allow_tf32", "expected_precision"), [(False, "ieee"), (True, "tf32")]
)
def test_set_tf32_new_api(monkeypatch, allow_tf32, expected_precision):
    fake_torch = NewTorch()
    monkeypatch.setattr(torch_setup, "torch", fake_torch)

    torch_setup.set_tf32(allow_tf32)

    assert fake_torch.backends.fp32_precision == expected_precision
    assert fake_torch.backends.cuda.matmul.fp32_precision == expected_precision
    assert fake_torch.backends.cudnn.fp32_precision == expected_precision
    assert fake_torch.backends.cudnn.conv.fp32_precision == expected_precision
    assert fake_torch.backends.cudnn.rnn.fp32_precision == expected_precision


def test_set_tf32_rejects_non_boolean_values():
    with pytest.raises(TypeError, match="allow_tf32 must be a bool"):
        torch_setup.set_tf32("false")
