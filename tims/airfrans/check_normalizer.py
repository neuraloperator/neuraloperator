import airfrans_dataset
from airfrans_dataset import SelectiveUnitGaussianNormalizer, SelectiveDataProcessor
from airfrans_dataset import AirfransDataset

# In AirfransDataset.__init__
input_encoder = SelectiveUnitGaussianNormalizer(dim=[0, 1, 3])
input_encoder.channels_to_normalize = [0, 1, 3]
input_encoder.fit(x_train[:, [0, 1, 3], :, :]) # Fit stats on the 3 channels

self._data_processor = SelectiveDataProcessor(
    in_normalizer=input_encoder,
    out_normalizer=output_encoder
)