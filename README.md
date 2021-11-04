# Optical Flow Estimation

## Prerequisites

If you like, you can download the original Multi-Human Optical Flow dataset from [here](https://humanflow.is.tue.mpg.de).
But note that you should use the data on the cluster, since the train/val/test splits have been modified and encrypted from the original dataset.


Download pre-trained PWC-Net models from [NVlabs/PWC-Net](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) and store them in `models/` directory.
You are encouraged to get familiar with the PWC-net.


Install Pytorch. Install dependencies using
```sh
pip3 install -r requirements.txt
```

## Training

For training our final RAFT model on Multi-Human Optical Flow datase use:
```sh
python main.py --data <path to dataset> --dataset humanflow --a raft --div-flow 1 --name base_raftfull --epochs 20 --lr 0.00005 --epoch-size 2000 -b 6
```

## Testing
For getting the final optical flow predictions on the test dataset, run
```sh
python test_humanflow.py --data <path to dataset> --dataset humanflow --arch raft --div-flow 1 --no-norm --pretrained <path to checkpoint>  --output-dir <path to output dir>
```
## Acknowledgements
We thank Clement Pinard for his github repository [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch). We use it as our code base. PWCNet is taken from [NVlabs/PWC-Net](https://github.com/NVlabs/PWC-Net). SPyNet implementation is taken from [sniklaus/pytorch-spynet](https://github.com/sniklaus/pytorch-spynet). Correlation module is taken from [ClementPinard/Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension).
