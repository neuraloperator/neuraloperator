import torch
import pytest
from ..fourier_continuation import FCLegendre, FCGram


@pytest.mark.parametrize("FC_class", [FCLegendre, FCGram])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("d", [3, 4])
def test_fourier_continuation(FC_class, dim: int, d: int):
    if dim == 1:
        batch_size = 3
        length_signal = 101
        add_pts = 50

        x = torch.linspace(0, 1, length_signal).repeat(batch_size, 1)
        f = torch.sin(16 * x) - torch.cos(8 * x)

        Extension = FC_class(d=d, n_additional_pts=add_pts)

        f_extend = Extension(f, dim=1)

        # Check shape
        assert f_extend.shape[-1] == f.shape[-1] + add_pts
        # Check values of original signal
        torch.testing.assert_close(f, f_extend[..., add_pts // 2 : -add_pts // 2])

    if dim == 2:
        batch_size = 3
        length_signal = 101
        add_pts = 50

        x = torch.linspace(0, 1, length_signal).view(1, length_signal, 1).repeat(batch_size, 1, length_signal)
        y = torch.linspace(0, 1, length_signal).view(1, 1, length_signal).repeat(batch_size, length_signal, 1)
        f = torch.sin(12 * x)  - torch.cos(14 * y) + 3*x*y
        
        Extension = FC_class(d=d, n_additional_pts=add_pts)

        f_extend = Extension(f, dim=2)

        # Check shape
        assert f_extend.shape[-1] == f.shape[-1] + add_pts
        assert f_extend.shape[-2] == f.shape[-2] + add_pts
        # Check values of original signal
        torch.testing.assert_close(
            f, f_extend[..., add_pts // 2 : -add_pts // 2, add_pts // 2 : -add_pts // 2]
        )

    if dim == 3:
        batch_size = 2
        length_signal = 81
        add_pts = 50

        # Create 3D grid
        x = torch.linspace(0, 1, length_signal).view(1, length_signal, 1, 1).repeat(batch_size, 1, length_signal, length_signal)
        y = torch.linspace(0, 1, length_signal).view(1, 1, length_signal, 1).repeat(batch_size, length_signal, 1, length_signal)
        z = torch.linspace(0, 1, length_signal).view(1, 1, 1, length_signal).repeat(batch_size, length_signal, length_signal, 1)

        # Create 3D function
        f = torch.exp(-2*z) + 2*z*x + torch.sin(12*x*y) + y*torch.sin(10*y*z) 

        Extension = FC_class(d=d, n_additional_pts=add_pts)

        f_extend = Extension(f, dim=3)

        # Check shape
        assert f_extend.shape[-1] == f.shape[-1] + add_pts
        assert f_extend.shape[-2] == f.shape[-2] + add_pts
        assert f_extend.shape[-3] == f.shape[-3] + add_pts
        # Check values of original signal
        torch.testing.assert_close(
            f,
            f_extend[
                ...,
                add_pts // 2 : -add_pts // 2,
                add_pts // 2 : -add_pts // 2,
                add_pts // 2 : -add_pts // 2,
            ],
        )


@pytest.mark.parametrize("FC_class", [FCLegendre, FCGram])
@pytest.mark.parametrize(
    "dim_tuple",
    [(1,), (2,), (1, 2), (1, 3), (2, 3), (-1,), (-2,), (-1, -2), (-1, -3), (1, -1)],
)
@pytest.mark.parametrize("d", [3, 4])
def test_fourier_continuation_tuple_dims(FC_class, dim_tuple, d: int):
    """
    Test Fourier continuation with tuple dimensions.

    Tests various combinations of positive and negative axis indices
    to ensure the extension works correctly along specific axes.
    """
    batch_size = 2
    add_pts = 30

    # Create a 4D tensor for comprehensive testing
    x = torch.randn(batch_size, 20, 25, 30)

    try:
        Extension = FC_class(d=d, n_additional_pts=add_pts)
    except FileNotFoundError as e:
        if FC_class.__name__ == "FCGram":
            pytest.skip(f"FCGram matrices not available: {e}")
        else:
            raise

    # Extend along specified dimensions
    f_extend = Extension(x, dim=dim_tuple)

    # Calculate expected shape
    expected_shape = list(x.shape)
    for dim in dim_tuple:
        # Convert negative indices to positive
        if dim < 0:
            dim = x.ndim + dim
        expected_shape[dim] += add_pts

    # Check shape
    assert f_extend.shape == tuple(
        expected_shape
    ), f"Expected shape {tuple(expected_shape)}, got {f_extend.shape}"

    # Test restrict functionality
    f_restricted = Extension.restrict(f_extend, dim=dim_tuple)

    # Check that restrict returns original shape
    assert (
        f_restricted.shape == x.shape
    ), f"Restrict failed: expected {x.shape}, got {f_restricted.shape}"

    # Check that restrict returns original values
    torch.testing.assert_close(
        x, f_restricted, msg="Restrict did not return original values"
    )
