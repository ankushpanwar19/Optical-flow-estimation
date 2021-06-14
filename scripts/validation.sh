# test
bsub -n 6 -W 50:00 -J "ssadat" -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" -o logs python test_humanflow.py --data /cluster/project/infk/hilliges/lectures/mp21/project6/dataset --dataset humanflow --output-dir results

# validation
# bsub -n 6 -W 50:00 -J "ssadat" -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" -o logs python test_humanflow.py --data /cluster/project/infk/hilliges/lectures/mp21/project6/dataset --dataset humanflow --output-dir validation-stats