import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_weights():
    # create an instance of MyModel
    model = MyModel()

    # print the initial weights of the model
    print("Initial weights:")
    for name, param in model.named_parameters():
        print(name, param.data)

    # generate some random input data
    x = torch.randn(1, 10)
    y = torch.tensor([1, 0], dtype=torch.float32).unsqueeze(0)

    # define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # run the input data through the model and update the weights
    for i in range(100):
        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print the updated weights of the model
    print("\nUpdated weights:")
    for name, param in model.named_parameters():
        print(name, param.data)

# run the test
test_weights()