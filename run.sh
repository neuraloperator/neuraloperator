# --logger_type wandb
# --data_path /datasets/car-pressure-data

# boyi3d:/workspace

$WANDB_API_KEY nvr-ai-algo/ahmed-body/fj41u3er

python train_ahmed.py --config config/FNOGNOAhmed.yaml 

ngc batch run --instance dgx1v.32g.2.norm --name 'ml-model.8gpu0' --image "nvidian/nvr-aialgo/geo-neuraloperator:latest" \
--workspace boyil:/workspace  --result /result --commandline "apt update; apt install tmux -y; pip install jupyter; sleep 167h" \
--datasetid 1606785:/datasets/ahmed-body --datasetid 4805448:/datasets/new_ahmed \
-p 8080

ngc batch run --instance dgx1v.16g.2.norm --name '8gpu1' --image "nvidia/pytorch:20.02-py3" \
--workspace boyil:/workspace  --result /result --commandline "apt update; apt install tmux -y; sleep 167h" \
--datasetid 1606785:/datasets/ahmed-body \
-p 8081

ngc batch run --instance dgx1v.32g.8.norm --name 'ml-model.8gpu2' --image "nvidian/nvr-aialgo/geo-neuraloperator:latest" \
--workspace boyil:/workspace  --result /result --commandline "apt update; apt install tmux -y; pip install jupyter; sleep 167h" \
--datasetid 1606785:/datasets/ahmed-body --datasetid 4805448:/datasets/new_ahmed \
-p 8082


#### initialization

ngc workspace mount --mode RW boyil ~/Desktop/projects/boyi/workspace/.

ngc batch exec --commandline bash 4805448

tmux new -s t1

pip install jupyter 

jupyter notebook --no-browser --NotebookApp.allow_origin='*' --port 8082 --allow-root 

ngc dataset convert --from-result 4805448 --desc 'new_ahmed' new_ahmed
# Dataset with ID: '1608682' created in ACE: 'nv-us-west-2'.
# ngc dataset convert --from-result 3668847 dataset_name

nohup python train_ahmed.py --config config/FNOGNOAhmed.yaml --logger_type wandb > log.txt 2>&1 &