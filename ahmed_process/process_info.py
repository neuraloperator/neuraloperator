import json
import torch
import os 

data_file = '/home/nikola/completed_results_combined.json'

with open(data_file, "r") as inp:
    results = json.load(inp)

for case in results.keys():
    path = 'data/case' + case

    if not os.path.exists(path):
        continue

    cp = results[case]['cd_history_p'][-1]
    cf = results[case]['cd_history_cf'][-1]
    l = results[case]['Length']
    w = results[case]['Width']
    h = results[case]['Height']
    gc = results[case]['GroundClearance']
    sa = results[case]['SlantAngle']
    fr = results[case]['FilletRadius']
    v = results[case]['Velocity']
    re = results[case]['Re (based on length)']

    info = {'c_p': cp, 
            'c_f': cf,
            'length': l,
            'width': w,
            'height': h,
            'ground_clearance': gc,
            'slant_angle': sa,
            'fillet_radius': fr,
            'velocity': v,
            're': re
            }

    torch.save(info, 'data/case' + case + '/info.pt')
    torch.save(torch.tensor([v], dtype=torch.float32), 'data/case' + case + '/inlet_velocity.pt')

    #print(case)
