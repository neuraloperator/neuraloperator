from tensorly.utils import DefineDeprecated
from neuralop.data.transforms import normalizers

warning_msg = "Warning: neuralop.datasets.output_encoder is deprecated and has been moved to neuralop.data.datasets.normalizers."
UnitGaussianNormalizer = DefineDeprecated(normalizers.UnitGaussianNormalizer, warning_msg)
MultipleFieldOutputEncoder = DefineDeprecated(normalizers.DictUnitGaussianNormalizer, warning_msg)
