# on cpu
# python3 main.py --arch raft --workers 8 --data /cluster/project/infk/hilliges/lectures/mp21/project6/dataset/ --dataset humanflow

# on gpu:
bsub -n 6 -W 50:00 -J "ssadat" -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" -o logs python main.py --data /cluster/project/infk/hilliges/lectures/mp21/project6/dataset --dataset humanflow --arch raft --div-flow 20 --name raft_experiment --epochs 50 --lr 0.0001 --batch_size 8 --weight-decay 0.00001 --epoch-size 2000

# test on gpu:
# bsub -n 1 -I -J "ssadat" -R "rusage[mem=4098, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python main.py --data /cluster/project/infk/hilliges/lectures/mp21/project6/dataset --dataset humanflow --arch raft --div-flow 20 --name raft_experiment --epochs 50 --lr 0.0001 --batch_size 8 --weight-decay 0.00001 --epoch-size 1 --workers 1
