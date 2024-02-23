import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TransformerNO
from neuralop.datasets import load_darcy_flow_small
from neuralop import LpLoss, H1Loss
from tqdm import tqdm

device = 'cuda:0'

train_loader, test_loaders, output_encoder = load_darcy_flow_small(
        n_train=1000, batch_size=32,
        test_resolutions=[16, 32],
        n_tests=[100, 50],
        test_batch_sizes=[32, 32],
        positional_encoding=False
)
model = TransformerNO(
        n_dim=2,
        in_channels=1,
        out_channels=1,
        encoder_hidden_channels=64,
        decoder_hidden_channels=64,
        encoder_num_heads=4,
        decoder_num_heads=16,
        encoder_n_layers=3,
        norm='layer_norm'
)
model = model.to(device)


optimizer = torch.optim.AdamW(model.parameters(),
                                lr=5e-4,
                                weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

l2loss = LpLoss(d=2, p=2, reductions='mean')
h1loss = H1Loss(d=2, reductions='mean')

train_loss = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}

# print('\n### MODEL ###\n', model)
# print('\n### OPTIMIZER ###\n', optimizer)
# print('\n### SCHEDULER ###\n', scheduler)
# print('\n### LOSSES ###')
# print(f'\n * Train: {train_loss}')
# print(f'\n * Test: {eval_losses}')

def model_wrapper(x, y):
        # adhoc code snippet for transforming input
        x = x.permute(0, 2, 3, 1)   # channel first to channel last
        nx, ny = x.shape[2], x.shape[1]
        input_pos_x, input_pos_y = torch.meshgrid(
                                [torch.linspace(0, 1, x.shape[1]),
                                 torch.linspace(0, 1, x.shape[2])])
        x = x.reshape(x.shape[0], -1, 1)
        input_pos = torch.stack([input_pos_x, input_pos_y], dim=-1).reshape(1, -1, 2).to(x.device)
        input_pos = input_pos.repeat(x.shape[0], 1, 1)
        y_pred = model(x, input_pos)
        y_pred = y_pred.reshape(y_pred.shape[0], ny, nx, -1).permute(0, 3, 1, 2)
        return y_pred

# training loop
n_epochs = 100
for epoch in range(n_epochs):
        model.train()
        with tqdm(total=len(train_loader)) as pbar:
                for data in train_loader:
                        x = data['x']
                        y = data['y']
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        y_pred = model_wrapper(x, y)

                        loss = train_loss(y_pred, y)
                        loss.backward()
                        optimizer.step()
                        # update desquery_pos.shape[0]cription
                        description = f'Epoch {epoch+1}/{n_epochs} |' \
                                      f' Loss: {loss.item():.4f} |' \
                                f' LR: {scheduler.get_last_lr()[0]:.4f}'
                        pbar.set_description(description)
        scheduler.step()

        if (epoch+1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                        vis_cache = {}
                        for (res, test_loader) in test_loaders.items():
                                y_lst = []
                                y_pred_lst = []

                                for data in test_loader:
                                        x = data['x']
                                        y = data['y']
                                        x, y = x.to(device), y.to(device)
                                        y_pred = model_wrapper(x, y)
                                        y_pred = output_encoder.decode(y_pred.cpu())
                                        y_lst.append(y.cpu())
                                        y_pred_lst.append(y_pred)

                                y = torch.cat(y_lst, dim=0)
                                y_pred = torch.cat(y_pred_lst, dim=0)
                                for l, loss in eval_losses.items():
                                        print(f'Test {res} | {l} loss: {loss(y_pred, y).item():.4f}')

                                vis_cache[res] = (y, y_pred)

for res, (y, y_pred) in vis_cache.items():
        # plt a random sample
        idx = torch.randint(0, y.shape[0], (1,)).item()
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(y[idx, 0, :, :].cpu().numpy())
        axs[0].set_title(f'Ground truth at resolution {res}')
        axs[1].imshow(y_pred[idx, 0, :, :].cpu().numpy())
        axs[1].set_title(f'Prediction at resolution {res}')
        plt.show()