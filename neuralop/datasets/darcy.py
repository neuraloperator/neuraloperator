from pathlib import Path
import torch

from .output_encoder import UnitGaussianNormalizer
from .tensor_dataset import TensorDataset
from .transforms import PositionalEmbedding2D
from .data_transforms import DefaultDataProcessor


def load_darcy_flow_small(
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    test_resolutions=[16, 32],
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
):
    """Loads a small Darcy-Flow dataset

    Training contains 1000 samples in resolution 16x16.
    Testing contains 100 samples at resolution 16x16 and
    50 samples at resolution 32x32.

    Parameters
    ----------
    n_train : int
    n_tests : int
    batch_size : int
    test_batch_sizes : int list
    test_resolutions : int list, default is [16, 32],
    grid_boundaries : int list, default is [[0,1],[0,1]],
    positional_encoding : bool, default is True
    encode_input : bool, default is False
    encode_output : bool, default is True
    encoding : 'channel-wise'
    channel_dim : int, default is 1
        where to put the channel dimension, defaults size is 1
        i.e: batch, channel, height, width

    Returns
    -------
    training_dataloader, testing_dataloaders

    training_dataloader : torch DataLoader
    testing_dataloaders : dict (key: DataLoader)
    """
    for res in test_resolutions:
        if res not in [16, 32]:
            raise ValueError(
                f"Only 32 and 64 are supported for test resolution, "
                f"but got test_resolutions={test_resolutions}"
            )
    path = Path(__file__).resolve().parent.joinpath("data")
    return load_darcy_pt(
        str(path),
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        test_resolutions=test_resolutions,
        train_resolution=16,
        grid_boundaries=grid_boundaries,
        positional_encoding=positional_encoding,
        encode_input=encode_input,
        encode_output=encode_output,
        encoding=encoding,
        channel_dim=channel_dim,
    )


def load_darcy_pt(
    data_path,
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    test_resolutions=[32],
    train_resolution=32,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
):
    """Load the Navier-Stokes dataset"""
    data = torch.load(
        Path(data_path).joinpath(f"darcy_train_{train_resolution}.pt").as_posix()
    )
    x_train = (
        data["x"][0:n_train, :, :].unsqueeze(channel_dim).type(torch.float32).clone()
    )
    y_train = data["y"][0:n_train, :, :].unsqueeze(channel_dim).clone()
    del data

    idx = test_resolutions.index(train_resolution)
    test_resolutions.pop(idx)
    n_test = n_tests.pop(idx)
    test_batch_size = test_batch_sizes.pop(idx)

    data = torch.load(
        Path(data_path).joinpath(f"darcy_test_{train_resolution}.pt").as_posix()
    )
    x_test = data["x"][:n_test, :, :].unsqueeze(channel_dim).type(torch.float32).clone()
    y_test = data["y"][:n_test, :, :].unsqueeze(channel_dim).clone()
    del data

    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(x_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
        #x_train = input_encoder.transform(x_train)
        #x_test = input_encoder.transform(x_test.contiguous())
    else:
        input_encoder = None

    if encode_output:
        if encoding == "channel-wise":
            reduce_dims = list(range(y_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        output_encoder.fit(y_train)
        #y_train = output_encoder.transform(y_train)
    else:
        output_encoder = None

    train_db = TensorDataset(
        x_train,
        y_train,
    )
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    test_db = TensorDataset(
        x_test,
        y_test,
    )
    test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loaders = {train_resolution: test_loader}
    for (res, n_test, test_batch_size) in zip(
        test_resolutions, n_tests, test_batch_sizes
    ):
        print(
            f"Loading test db at resolution {res} with {n_test} samples "
            f"and batch-size={test_batch_size}"
        )
        data = torch.load(Path(data_path).joinpath(f"darcy_test_{res}.pt").as_posix())
        x_test = (
            data["x"][:n_test, :, :].unsqueeze(channel_dim).type(torch.float32).clone()
        )
        y_test = data["y"][:n_test, :, :].unsqueeze(channel_dim).clone()
        del data
        #if input_encoder is not None:
            #x_test = input_encoder.transform(x_test)

        test_db = TensorDataset(
            x_test,
            y_test,
        )
        test_loader = torch.utils.data.DataLoader(
            test_db,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
        test_loaders[res] = test_loader 

    
    if positional_encoding:
        pos_encoding = PositionalEmbedding2D(grid_boundaries=grid_boundaries)
    else:
        pos_encoding = None
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
        positional_encoding=pos_encoding
    )
    return train_loader, test_loaders, data_processor
def load_darcy_421_5k(data_root: Path, 
                        n_train, n_test,
                        batch_size, test_batch_size,
                        sub=1,
                        grid_boundaries=[[0,1],[0,1]],
                        positional_encoding=True,
                        encode_input=False,
                        encode_output=True,
                        encoding='channel-wise', 
                        channel_dim=1):
    """
    Dataloader for Nik's 421 5K darcy dataset

    data is saved as a torch .pt archive
    """
    if isinstance(data_root, str):
        data_root = Path(data_root)
    

    train_data = torch.load(data_root / "darcy_train_421.pt")
    a = train_data['x']
    u = train_data['y']

    x_train = a[0:n_train, ::sub, ::sub].unsqueeze(channel_dim)
    y_train = u[0:n_train, ::sub, ::sub].unsqueeze(channel_dim)
    del train_data

    if n_test > 0:
        test_data = torch.load(data_root / "darcy_test_421.pt")
        a = test_data['x']
        u = test_data['y']
        x_test = a[:n_test, ::sub, ::sub].unsqueeze(channel_dim)
        y_test = u[:n_test, ::sub, ::sub].unsqueeze(channel_dim)
    del test_data
    train_resolution = x_train.shape[-1]

    # this part is the same as in load_darcy_pt
    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(x_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
        #x_train = input_encoder.transform(x_train)
        #x_test = input_encoder.transform(x_test.contiguous())
    else:
        input_encoder = None

    if encode_output:
        if encoding == "channel-wise":
            reduce_dims = list(range(y_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        output_encoder.fit(y_train)
        #y_train = output_encoder.transform(y_train)
    else:
        output_encoder = None

    if positional_encoding:
        pos_enc = PositionalEmbedding2D(grid_boundaries)
    else:
        pos_enc = None

    data_proc = DefaultDataProcessor(in_normalizer=input_encoder, out_normalizer=output_encoder, positional_encoding=pos_enc)
    
    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size, 
        # shuffle=True,
        shuffle=False,
        num_workers=0, 
        pin_memory=True, 
        persistent_workers=False
    )

    if n_test > 0:
        test_db = TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_db,
                                                batch_size=test_batch_size, shuffle=False,
                                                num_workers=0, pin_memory=True, persistent_workers=False)
        test_loaders =  {train_resolution: test_loader}
    else: 
        test_loaders = None

    return train_loader, train_db, test_loaders, data_proc


def loader_to_residual_db(model, data_processor, loader, device, train_val_split=True):
    """train_db_to_residual_train_db converts a dataset of x: a(x), y: u(x) to 
    x: a(x), y: G(a,x) - u(x)"""
    error_list = []
    x_list = []
    model = model.to(device)
    model.eval()
    data_processor.train = False # unnormalized y
    data_processor = data_processor.to(device)
    for idx, sample in enumerate(loader):
        #print(f"input pre preproc {sample['y'].mean()=}")
        sample = data_processor.preprocess(sample)
        #print(f"input post preproc {sample['y'].mean()=}")
        out = model(**sample)
        #print(f"output pre postproc {out.mean()=}")
        out, sample = data_processor.postprocess(out, sample) # unnormalize output
        #print(f"output post postproc {out.mean()=}")
        #print(f"output post postproc {sample['y'].mean()=}")

        x_list.append(sample['x'].to("cpu"))
        error = (out-sample['y']).detach().to("cpu") # detach, otherwise residual carries gradient of model weight
        # error is unnormalized here
        error_list.append(error)
        
        del sample, out
    errors = torch.cat(error_list, axis=0)
    xs = torch.cat(x_list, axis=0) # check this
    print(f"{errors.shape=} {xs.shape=}")
    
    residual_encoder = UnitGaussianNormalizer()
    print(f"{residual_encoder.mean=}")
    print(f"{torch.mean(errors)=}")
    print(f"{torch.var(errors)=}")
    residual_encoder.fit(errors)
    print(f"{residual_encoder.mean=}")
    print(f"{residual_encoder.std=}")

    #errors = residual_encoder.transform(errors)
    
    # positional encoding and normalization already applied to X values
    residual_data_processor = DefaultDataProcessor(in_normalizer=None,
                                                   out_normalizer=residual_encoder, 
                                                   positional_encoding=None)
    residual_data_processor.train = True

    if train_val_split:
        val_start = int(0.8 * xs.shape[0])

        residual_train_db = TensorDataset(x=xs[:val_start], y=errors[:val_start])
        residual_val_db = TensorDataset(x=xs[val_start:], y=errors[val_start:])
    else:
        residual_val_db = None
    return residual_train_db, residual_val_db, residual_data_processor