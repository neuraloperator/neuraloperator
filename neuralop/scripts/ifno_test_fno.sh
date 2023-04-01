TASK_NAME=ifno_batch_script_test_fno

for BF in 0
do
    for SCALE in 0
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.32g.4.norm \
            --image nvcr.io/nvidian/nvr-aialgo/fly-incremental:zoo_latest \
            --result /results \
            --workspace 6Ubcqvn_Rn6uKFJw4ijJdw:/ngc_workspace \
            --datasetid 23145:/dataset \
            --team nvr-aialgo \
            --port 6006 --port 1234 --port 8888 \
            --commandline "bash -c '\
                sh /ngc_workspace/jiawei/set_wandb.sh; \
                cd /workspace; \
                git clone ; \
                cd /workspace/neuraloperator/neuralop/models/tests; \
                git checkout robert-test-incremental; \
                cp -r /ngc_workspace/jiawei/projects/ifno/data /workspace/fly-incremental/data; \
                python test_darcy_baseline.py; \
                python test_darcy_incremental.py; \
                python test_darcy_incremental_loss_gap.py; \
                python test_darcy_incremental_resolution.py ; \
            '"
    done
done
