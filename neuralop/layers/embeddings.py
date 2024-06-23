from tensorly.utils import DefineDeprecated
from neuralop.data.transforms import positional_embeddings

warning_msg = "Warning: neuralop.layers.embeddings is deprecated and has been moved to neuralop.data.transforms.positional_embeddings."
PositionalEmbedding2D = DefineDeprecated(positional_embeddings.GridEmbedding2D, warning_msg)
RotaryEmbedding = DefineDeprecated(positional_embeddings.RotaryEmbedding2D, warning_msg)
PositionalEmbedding = DefineDeprecated(positional_embeddings.SinusoidalEmbedding2D, warning_msg)