from pathlib import Path
import torch

def load_burgers(data_path, n_train, n_test, batch_train=32, batch_test=100, 
                 time=1, grid=[0,1]):

    data_path = Path(data_path).joinpath('burgers.pt').as_posix()
    data = torch.load(data_path)

    x_train = data[0:n_train,:,0]
    x_test = data[n_train:(n_train + n_test),:,0]

    y_train = data[0:n_train,:,time]
    y_test = data[n_train:(n_train + n_test),:,time]

    s = x_train.size(-1)
    
    if grid is not None:
        grid = torch.linspace(grid[0], grid[1], s + 1)[0:-1].view(1,-1)

        grid_train = grid.repeat(n_train, 1)
        grid_test = grid.repeat(n_test, 1)

        x_train = torch.cat((x_train.unsqueeze(1), grid_train.unsqueeze(1)), 1)
        x_test = torch.cat((x_test.unsqueeze(1), grid_test.unsqueeze(1)), 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_train, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_test, shuffle=False)

    return train_loader, test_loader