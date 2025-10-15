'''from neuralop.data.datasets.ot_datamodule import OTDataModule

if __name__ == "__main__":
    data_module = OTDataModule(
            root_dir="D:/python_code/data/car-pressure-data",
            item_dir_name="data",
            n_total=2,
        )'''
from neuralop.data.datasets.car_ot_dataset import CarOTDataset

if __name__ == "__main__":
    data_module = CarOTDataset(
            root_dir="D:/python_code/data/car-pressure-data",
            n_train=2,
            n_test=1,
        )