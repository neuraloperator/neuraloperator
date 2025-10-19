import pytest
import torch

try:
    import torch_harmonics
except ModuleNotFoundError:
    pytest.skip(
        "Skipping because torch_harmonics is not installed", allow_module_level=True
    )

from ..local_no_block import LocalNOBlocks


@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_LocalNOBlock_resolution_scaling_factor(n_dim):
    """Test LocalNOBlocks with upsampled or downsampled outputs"""
    max_n_modes = [8, 8, 8]
    n_modes = [4, 4, 4]

    size = [10] * 3
    channel_mlp_dropout = 0
    channel_mlp_expansion = 0.5
    channel_mlp_skip = "linear"
    block = LocalNOBlocks(
        3,
        4,
        default_in_shape=tuple(size[:n_dim]),
        n_modes=max_n_modes[:n_dim],
        max_n_modes=max_n_modes[:n_dim],
        n_layers=1,
        diff_layers=[True],
        disco_layers=[(n_dim == 2)],
    )

    assert block.convs[0].n_modes[:-1] == max_n_modes[: n_dim - 1]
    assert block.convs[0].n_modes[-1] == max_n_modes[n_dim - 1] // 2 + 1

    block.n_modes = n_modes[:n_dim]
    assert block.convs[0].n_modes[:-1] == n_modes[: n_dim - 1]
    assert block.convs[0].n_modes[-1] == n_modes[n_dim - 1] // 2 + 1

    block.n_modes = max_n_modes[:n_dim]
    assert block.convs[0].n_modes[:-1] == max_n_modes[: n_dim - 1]
    assert block.convs[0].n_modes[-1] == max_n_modes[n_dim - 1] // 2 + 1

    # Downsample outputs
    block = LocalNOBlocks(
        3,
        4,
        n_modes[:n_dim],
        default_in_shape=tuple(size[:n_dim]),
        n_layers=1,
        diff_layers=[True],
        disco_layers=[(n_dim == 2)],
        resolution_scaling_factor=0.5,
        use_channel_mlp=True,
        channel_mlp_dropout=channel_mlp_dropout,
        channel_mlp_expansion=channel_mlp_expansion,
        channel_mlp_skip=channel_mlp_skip,
    )

    x = torch.randn(2, 3, *size[:n_dim])
    res = block(x)
    assert list(res.shape[2:]) == [m // 2 for m in size[:n_dim]]

    # Upsample outputs
    block = LocalNOBlocks(
        3,
        4,
        n_modes[:n_dim],
        default_in_shape=tuple(size[:n_dim]),
        n_layers=1,
        diff_layers=[True],
        disco_layers=[(n_dim == 2)],
        resolution_scaling_factor=2,
        use_channel_mlp=True,
        channel_mlp_dropout=channel_mlp_dropout,
        channel_mlp_expansion=channel_mlp_expansion,
        channel_mlp_skip=channel_mlp_skip,
    )

    x = torch.randn(2, 3, *size[:n_dim])
    res = block(x)
    assert res.shape[1] == 4  # Check out channels
    assert list(res.shape[2:]) == [m * 2 for m in size[:n_dim]]


@pytest.mark.parametrize("norm", ["instance_norm", "ada_in", "group_norm"])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_LocalNOBlock_norm(norm, n_dim):
    """Test LocalNOBlock with normalization"""
    modes = (8, 8, 8)
    size = [10] * 3
    channel_mlp_dropout = 0
    channel_mlp_expansion = 0.5
    channel_mlp_skip = "linear"
    ada_in_features = 4
    block = LocalNOBlocks(
        3,
        4,
        modes[:n_dim],
        default_in_shape=tuple(size[:n_dim]),
        n_layers=1,
        diff_layers=[True],
        disco_layers=[(n_dim == 2)],
        use_channel_mlp=True,
        norm=norm,
        ada_in_features=ada_in_features,
        channel_mlp_dropout=channel_mlp_dropout,
        channel_mlp_expansion=channel_mlp_expansion,
        channel_mlp_skip=channel_mlp_skip,
    )

    if norm == "ada_in":
        embedding = torch.randn(ada_in_features)
        block.set_ada_in_embeddings(embedding)

    x = torch.randn(2, 3, *size[:n_dim])
    res = block(x)
    assert list(res.shape[2:]) == size[:n_dim]


@pytest.mark.parametrize("local_no_skip", ["linear", None])
@pytest.mark.parametrize("channel_mlp_skip", ["linear", None])
def test_LocalNOBlock_skip_connections(local_no_skip, channel_mlp_skip):
    """Test LocalNOBlocks with different skip connection options including None"""
    modes = (8, 8, 8)
    size = [10, 10, 10]

    # Test with channel MLP enabled
    block = LocalNOBlocks(
        3,
        4,
        modes,
        default_in_shape=tuple(size),
        n_layers=2,
        local_no_skip=local_no_skip,
        channel_mlp_skip=channel_mlp_skip,
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        channel_mlp_dropout=0.0,
        diff_layers=[True, True],
        disco_layers=[False, False],
    )

    x = torch.randn(2, 3, *size)
    res = block(x)

    # Check output shape
    assert res.shape == (2, 4, *size)

    # Test with channel MLP disabled
    block_no_mlp = LocalNOBlocks(
        3,
        4,
        modes,
        default_in_shape=tuple(size),
        n_layers=2,
        local_no_skip=local_no_skip,
        channel_mlp_skip=channel_mlp_skip,
        use_channel_mlp=False,
        diff_layers=[True, True],
        disco_layers=[False, False],
    )

    res_no_mlp = block_no_mlp(x)
    assert res_no_mlp.shape == (2, 4, *size)
