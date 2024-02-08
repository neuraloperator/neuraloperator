import gdown

def load_fno_model(url, file_name, config):
    gdown.download(url, file_name, quiet=False)
    model = TFNO(n_modes=config["n_modes"],
                 hidden_channels=config["hidden_channels"],
                 projection_channels=config["projection_channels"],
                 factorization='tucker', rank=0.42)
    model.load_state_dict(torch.load("file_name"))
    return model

def load_base_model_encoder():
    encoder = UnitGaussianNormalizer(mean=torch.Tensor([[[0.0351]]]), std=torch.Tensor([[[0.0220]]], eps=0.00001))
    return encoder

def load_residual_model_encoder():
    encoder = UnitGaussianNormalizer(mean=torch.Tensor([[[0.000074001]]]), std=torch.Tensor([[[0.0002]]], eps=0.00001))
    return encoder

def get_darcy_loader_data(datax, datay, batch_size, shuffle, encode_output,
                            grid_boundaries=[[0,1],[0,1]],
                            positional_encoding=True,
                            encode_input=False,
                            encoding='channel-wise',
                            channel_dim=1):
    datax_tensor = torch.Tensor(datax)
    datay_tensor = torch.Tensor(datay)
    if positional_encoding:
        x_train = datax_tensor.unsqueeze(1).clone()
        y_train = datay_tensor.unsqueeze(1).clone()
    else:
        x_train = datax_tensor.clone()
        y_train = datay_tensor.clone()
    del datax
    del datay

    if encode_input:
        if encoding == 'channel-wise':
            reduce_dims = list(range(x_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(x_train, reduce_dim=reduce_dims)
        x_train = input_encoder.encode(x_train)
    else:
        input_encoder = None

    if encode_output:
        if encoding == 'channel-wise':
            reduce_dims = list(range(y_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(y_train, reduce_dim=reduce_dims)
        y_train = output_encoder.encode(y_train)
    else:
        output_encoder = None

    train_db = TensorDataset(x_train, y_train, transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
    train_loader = torch.utils.data.DataLoader(train_db,
                                               batch_size=batch_size, shuffle=shuffle,
                                               num_workers=0, pin_memory=True, persistent_workers=False)

    return train_loader, output_encoder

def download_darcy421_data(darcy_x_filename, darcy_y_filename):
    gdown.download("https://drive.google.com/file/d/1uQv1geg3HTZjtCS0reh9TMW0EBvvm7xm/view?usp=drive_link", darcy_x_filename, quiet=False)
    gdown.download("https://drive.google.com/file/d/1zIo5wVhvVJOaufXlJ8uDVpXm1qncqseL/view?usp=drive_link", darcy_y_filename, quiet=False)
    return

# split into calibration and test loaders
def get_calib_test_loaders(darcy_x_filename, darcy_y_filename, batch_size):
    darcy_x = scipy.io.loadmat(darcy_x_filename)["lognormal"]
    darcy_y = scipy.io.loadmat(darcy_y_filename)["darcylog"]
    x_calib = darcy_x[:,:,:512]
    y_calib = darcy_y[:,:,:512]
    x_test = darcy_x[:,:,512:]
    y_test = darcy_y[:,:,512:]
    calib_loader, _ = get_darcy_loader_data(x_calib, y_calib, batch_size, shuffle=False, encode_output=False)
    test_loader, _ = get_darcy_loader_data(x_test, y_test, batch_size, shuffle=False, encode_output=False)
    return calib_loader, test_loader

    



