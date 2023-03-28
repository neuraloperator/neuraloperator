import time
from torchsummary import summary
from neuralop.models.tfno import UNO
import torch

def test_UNO():
        layer_configs = [{"out_channels":20, "n_modes" : [5,5], "res_scaling" :[0.5,0.5] },\
                        {"out_channels":20, "n_modes" : [5,5], "res_scaling" :[1,1] },\
                        {"out_channels":20, "n_modes" : [5,5], "res_scaling" :[1,1] },\
                        {"out_channels":20, "n_modes" : [5,5], "res_scaling" :[2,2] },\
                        {"out_channels":10, "n_modes" : [5,5], "res_scaling" :[2,2] },\
                                ]
        horizontal_skips_map ={4:0,3:1}
        model = UNO(3,3,5,layer_configs = layer_configs, horizontal_skips_map = horizontal_skips_map, n_layers = 5, domain_padding = 0.2, output_scale_factor = 2)
        summary(model,(3,20,20))

        t1 = time.time()
        in_data = torch.randn(10,3,20,20).to('cuda')
        out = model(in_data)
        t = time.time() - t1
        print(f'Output of size {out.shape} in {t}.')

        loss = out.sum()
        loss.backward()
        n_unused_params = 0

        for name,param in model.named_parameters():
                if param.grad is not None:
                        print("Used-->",name)
                else:
                        print("Unused XXX---", name)
                        n_unused_params += 1

        assert n_unused_params == 0, f'{n_unused_params} parameters were unused!'