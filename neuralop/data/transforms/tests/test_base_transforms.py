import torch
from ..base_transforms import Transform, CompositeTransform


class AddOne(Transform):
    def transform(self, x):
        return x + 1.0

    def inverse_transform(self, x):
        return x - 1.0


class TimesTwo(Transform):
    def transform(self, x):
        return x * 2.0

    def inverse_transform(self, x):
        return x / 2.0


class AddThree(Transform):
    def transform(self, x):
        return x + 3.0

    def inverse_transform(self, x):
        return x - 3.0


class NoToTransform:
    """Helper that deliberately has no ``to`` method."""

    def transform(self, x):
        return x + 1.0

    def inverse_transform(self, x):
        return x - 1.0


def test_CompositeTransform_construct_and_transform():
    composite = CompositeTransform([AddOne(), TimesTwo()])
    x = torch.tensor([5.0])
    out = composite.transform(x)
    # add 1 then multiply by 2
    assert torch.allclose(out, torch.tensor([12.0]))


def test_CompositeTransform_inverse_is_identity():
    composite = CompositeTransform([AddOne(), TimesTwo(), AddThree()])
    x = torch.tensor([4.0])
    out = composite.inverse_transform(composite.transform(x))
    assert torch.allclose(out, x)


def test_CompositeTransform_to_keeps_no_to_transform():
    notrans = NoToTransform()
    composite = CompositeTransform([AddOne(), notrans])
    result = composite.to("cpu")
    assert result is composite
    assert len(composite.transforms) == 2
    assert composite.transforms[1] is notrans
    # kept transform is unchanged and still callable
    assert torch.allclose(
        composite.transforms[1].transform(torch.tensor([1.0])), torch.tensor([2.0])
    )
