from pathlib import Path
from typing import Literal, List
import yaml

import torch

from the_well.data import WellDataset
from the_well.utils.download import well_download


from ..transforms.normalizers import UnitGaussianNormalizer
from ..transforms.the_well_data_processors import TheWellDataProcessor

class TheWellDataset:
    """__init__ _summary_
        Base Class for TheWell [1]_ datasets
        
        Parameters
        ----------
        root_dir : Path
            shared root path at which to download all TheWell datasets
        well_dataset_name : str
            name of the dataset to download
        n_train : int
            _description_
        n_test : int
            _description_
        download : bool, optional
            _description_, by default True
        
        References
        ----------
        .. [1] : Ohana, R., McCabe, M., Meyer, L., Morel, R., Agocs, F., Benitez, M., Berger, M.,
            Burkhart, B., Dalziel, S., Fielding, D., Fortunato, D., Goldberg, J., Hirashima, K., Jiang, Y.,
            Kerswell, R., Maddu, S., Miller, J., Mukhopadhyay, P., Nixon, S., Shen, J., Watteaux, R., 
            Blancard, B., Rozet, F., and Parker, L., and Cranmer, M., and Ho, S. (2024).
            The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning. 
            NeurIPS 2024, https://openreview.net/forum?id=00Sx577BT3. 
        """
    def __init__(self,
                 root_dir: Path, 
                 well_dataset_name : str,
                 train_task: Literal['next_step']='next_step',
                 eval_tasks: List[Literal['next_step', 'rollout']]=['next_step'],
                 n_steps_input: int=1,
                 n_steps_output: int=1,
                 download: bool=True,
                 first_only: bool=True,
                 ):
        
        base_path = root_dir / f"datasets/{well_dataset_name}/data"

        if download:
            for split in ['train', 'test', 'valid']:
                data_path = base_path / split
                if not data_path.exists():
                    well_download(root_dir,
                                dataset=well_dataset_name,
                                split=split,
                                first_only=first_only,
                                )
        
        if train_task == "next_step":
            self._train_db = WellDataset(path=str(base_path / "train"),
                                            n_steps_input=n_steps_input,
                                            n_steps_output=n_steps_output,
                                            full_trajectory_mode=False,
                                            return_grid=False,
                                            use_normalization=False)
        else:
            raise NotImplementedError(f"no training set available for training task {train_task}. Choose from ['next_step']")
    
        self._test_dbs = {}
        
        if "next_step" in eval_tasks:
            self._test_dbs["next_step"] = WellDataset(path=str(base_path / "test"),
                                            n_steps_input=n_steps_input,
                                            n_steps_output=n_steps_output,
                                            return_grid=False,
                                            use_normalization=False)
        if "autoregression" in eval_tasks:
            self._test_dbs["autoregression"] = WellDataset(path=str(base_path / "test"),
                                            full_trajectory_mode=True,
                                            return_grid=False,
                                            use_normalization=False)
        
        stats_path = base_path / "../stats.yaml"
        with open(stats_path, "r") as f:
            stats = yaml.safe_load(f)
            from pprint import pprint
            pprint(stats)

        
        dataset_field_names = self._train_db.field_names
        pprint(dataset_field_names)

        # store channel-specific means for fields encoding the components of each physical variable
        channel_means = []
        channel_stds = []

        # keep track of constant field statistics separately
        const_means = []
        const_stds = []

        # Loop through fields separately: const, vector and 
        # tensor fields need to be handled differently. 

        
        # scalar fields have scalar means
        for field_name in dataset_field_names[0]:
            channel_means.append(stats['mean'][field_name])
            channel_stds.append(stats['std'][field_name])
        
        # vector-valued fields have vector means, 1 degree of nesting to unpack
        indiv_vector_fields = dataset_field_names[1]
        vector_fields = set(["_".join(x.split("_")[:-1]) for x in indiv_vector_fields])
        for field_name in vector_fields:
            channel_means.extend(stats['mean'][field_name])
            channel_stds.extend(stats['std'][field_name])

        # tensor-valued fields have tensor means, 2 degrees of nesting to unpack
        indiv_tensor_fields = dataset_field_names[2]
        tensor_fields = set(["_".join(x.split("_")[:-1]) for x in indiv_tensor_fields])
        for field_name in tensor_fields:
            channel_means.extend([x for xs in stats['mean'][field_name] for x in xs])
            channel_stds.extend([x for xs in stats['std'][field_name] for x in xs])
        
        # add a batch dimension and a dummy time dimension : final shape (1, channels, 1)
        # all reshaping/channel flattening will be performed after normalizing
        channel_means = torch.tensor(channel_means).unsqueeze(0).unsqueeze(-1)
        channel_stds = torch.tensor(channel_stds).unsqueeze(0).unsqueeze(-1)

        # unsqueeze means and stds along the spatial dimensions: final shape (1, channels, 1, 1, ... 1) 
        for _ in range(self.train_db.metadata.n_spatial_dims):
            channel_means = channel_means.unsqueeze(-1)
            channel_stds = channel_stds.unsqueeze(-1)

        data_normalizer = UnitGaussianNormalizer(mean=channel_means, std=channel_stds)

        # if there are any constant fields, provide a constant normalizer
        if const_means:
            const_means = torch.tensor(const_means).unsqueeze(0)
            const_stds = torch.tensor(const_stds).unsqueeze(0)

            # unsqueeze means and stds along the spatial dimensions
            for _ in range(self.train_db.metadata.n_spatial_dims):
                const_means = const_means.unsqueeze(-1)
                const_stds = const_stds.unsqueeze(-1)

            const_fields_normalizer = UnitGaussianNormalizer(mean=const_means, std=const_stds)
        else:
            const_fields_normalizer = None

        self._data_processor = TheWellDataProcessor(data_normalizer=data_normalizer, 
                                                    const_normalizer=const_fields_normalizer,
                                                    n_steps_input=n_steps_input, n_steps_output=n_steps_output, 
                                                    time_as_channels=True)
        
    @property
    def train_db(self):
        return self._train_db
    
    @property
    def test_dbs(self):
        return self._test_dbs
    
    @property
    def data_processor(self):
        return self._data_processor
    
class ActiveMatterDataset(TheWellDataset):
    def __init__(self, 
                 root_dir, 
                 train_task = 'next_step', 
                 eval_tasks = ['next_step'], 
                 n_steps_input: int=1,
                 n_steps_output: int=1,
                 download:bool = True, 
                 first_only = True):
        super().__init__(root_dir, well_dataset_name="active_matter", 
                         train_task=train_task,
                         eval_tasks=eval_tasks,
                         n_steps_input=n_steps_input,
                         n_steps_output=n_steps_output,
                         download=download,
                         first_only=first_only)

class MHD64Dataset(TheWellDataset):
    def __init__(self, 
                 root_dir, 
                 train_task = 'next_step', 
                 eval_tasks = ['next_step'], 
                 download = True, 
                 first_only = True):
        super().__init__(root_dir, well_dataset_name="MHD_64", 
                         train_task=train_task,
                         eval_tasks=eval_tasks,
                         download=download,
                         first_only=first_only)