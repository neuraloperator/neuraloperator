from neuralop.data.datasets import load_darcy_flow_small

# This should now return the loaders immediately without trying to download
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000, 
    n_tests=[100],
    batch_size=32, 
    test_batch_sizes=[32],
    test_resolutions=[16,32],
    data_root='/home/timm/Projects/PIML/neuraloperator/neuralop/data/datasets/data'
)

train_loader

print("Train loader size:", len(train_loader))