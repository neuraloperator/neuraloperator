from tensorly.utils import DefineDeprecated

warning_msg = "Warning: neuralop.datasets.output_encoder is deprecated and has been moved to neuralop.data.datasets.normalizers."
UnitGaussianNormalizer = DefineDeprecated('neuralop.data.transforms.normalizers.UnitGaussianNormalizer', warning_msg)
MultipleFieldOutputEncoder = DefineDeprecated('neuralop.data.transforms.normalizers.DictUnitGaussianNormalizer', warning_msg)
