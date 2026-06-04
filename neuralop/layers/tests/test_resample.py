from ..resample import resample
import torch


def test_resample():
    a = torch.randn(10, 20, 40, 50)

    res_scale = [2, 3]
    axis = [-2, -1]

    b = resample(a, res_scale, axis)
    assert b.shape[-1] == 3 * a.shape[-1] and b.shape[-2] == 2 * a.shape[-2]

    a = torch.randn((10, 20, 40, 50, 60))

    res_scale = [0.5, 3, 4]
    axis = [-3, -2, -1]
    b = resample(a, res_scale, axis)

    assert (
        b.shape[-1] == 4 * a.shape[-1]
        and b.shape[-2] == 3 * a.shape[-2]
        and b.shape[-3] == int(0.5 * a.shape[-3])
    )


def test_resample_fourier_mode():
    test_cases = [
        ((8,), (12,), (2,)),
        ((5,), (7,), (2,)),
        ((6, 8), (10, 12), (1, -2)),
        ((5, 6, 8), (7, 9, 12), (2, 1, 2)),
        ((5, 6, 7, 8), (7, 9, 10, 12), (1, -1, 2, 2)),
    ]

    for spatial_shape, output_shape, frequencies in test_cases:
        grids = torch.meshgrid(
            *[torch.arange(n) for n in spatial_shape], indexing="ij"
        )
        phase = sum(
            f * grid / n for f, grid, n in zip(frequencies, grids, spatial_shape)
        )
        output_grids = torch.meshgrid(
            *[torch.arange(n) for n in output_shape], indexing="ij"
        )
        output_phase = sum(
            f * grid / n for f, grid, n in zip(frequencies, output_grids, output_shape)
        )

        x_complex = torch.exp(2j * torch.pi * phase).reshape(1, 1, *spatial_shape)
        expected_complex = torch.exp(2j * torch.pi * output_phase).reshape(
            1, 1, *output_shape
        )
        y_complex = resample(
            x_complex, 1.0, list(range(2, x_complex.ndim)), output_shape=output_shape
        )

        assert y_complex.shape == (1, 1, *output_shape)
        assert y_complex.dtype == x_complex.dtype
        assert torch.is_complex(y_complex)
        torch.testing.assert_close(
            y_complex, expected_complex, atol=1e-5, rtol=1e-5
        )

        if len(spatial_shape) > 2:
            x_real = torch.cos(2 * torch.pi * phase).reshape(1, 1, *spatial_shape)
            expected_real = torch.cos(2 * torch.pi * output_phase).reshape(
                1, 1, *output_shape
            )
            y_real = resample(
                x_real, 1.0, list(range(2, x_real.ndim)), output_shape=output_shape
            )

            assert y_real.shape == (1, 1, *output_shape)
            assert y_real.dtype == x_real.dtype
            assert not torch.is_complex(y_real)
            torch.testing.assert_close(y_real, expected_real, atol=1e-5, rtol=1e-5)
