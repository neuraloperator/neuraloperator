import torch
from ..codano import CODANO
import pytest

def test_CODANO():
    input_variable_codimension = 1
    output_variable_codimension = 1
    hidden_variable_codimension = 1
    lifting_variable_codimension = 1
    use_positional_encoding = True
    n_variables = 3
    positional_encoding_dim = 4
    positional_encoding_modes = [8, 8]
    n_layers = 3
    n_heads = [2,2, 1]
    n_modes = [[32,32], [32,32], [32,32]]
    attention_scalings = [0.5,0.5, 0.5]
    scalings = [[0.5,0.5],[1,1], [2,2]]
    use_horizontal_skip_connection = True
    horizontal_skips_map = {2:0}

    model = CODANO(
        input_variable_codimension=input_variable_codimension,
        output_variable_codimension=output_variable_codimension,
        hidden_variable_codimension=hidden_variable_codimension,
        lifting_variable_codimension=lifting_variable_codimension,
        use_positional_encoding=use_positional_encoding,
        positional_encoding_dim=positional_encoding_dim,
        positional_encoding_modes=positional_encoding_modes,
        use_horizontal_skip_connection=use_horizontal_skip_connection,
        horizontal_skips_map=horizontal_skips_map,
        n_variables=n_variables,
        n_layers=n_layers,
        n_heads=n_heads,
        n_modes=n_modes,
        attention_scalings=attention_scalings,
        scalings=scalings
    )

    in_data = torch.randn(2, n_variables, 32, 32)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [2, n_variables, 32, 32]



