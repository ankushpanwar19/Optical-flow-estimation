# Humanflow2
This is an official repository

*Anurag Ranjan, David T. Hoffmann, Dimitrios Tzionas, Siyu Tang, Javier Romero, and Michael J. Black.* Learning Multi-Human Optical Flow. IJCV 2019.


[[Project Page]](https://humanflow.is.tue.mpg.de/)
[[Arxiv]](https://arxiv.org/abs/1910.11667)

## Prerequisites
Download the modified version for Machine Perception project on the Leonhard cluster, via 
```
/cluster/project/infk/hilliges/lectures/mp20/project6/
```

If you like, you can download the original Multi-Human Optical Flow dataset from [here](https://humanflow.is.tue.mpg.de).
But note that you should use the data on the cluster, since the train/val/test splits have been modified and encrypted from the original dataset.


Download pre-trained PWC-Net models from [NVlabs/PWC-Net](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) and store them in `models/` directory.
You are encouraged to get familiar with the PWC-net.


Install Pytorch. Install dependencies using
```sh
pip3 install -r requirements.txt
```

If there are issues with the correlation module, compile it from source - [ClementPinard/Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension).
## Training
Note that you are expected to read the code and understand the arguments/hyper-parameters.

For finetuning PWC-Net on Multi-Human Optical Flow dataset use:

```sh
python main.py PATH_TO_DATASET --dataset humanflow -a pwc --div-flow 20 -b8 -j8 --lr LEARNING_RATE --name NAME_OF_EXPERIMENT
```
## Testing

To test PWC-Net trained on Multi-Human Optical Flow dataset, use
```sh
python test_humanflow.py PATH_TO_DATASET --dataset humanflow --arch pwc --div-flow 20 --no-norm  --pretrained pretrained/pwc_MHOF.pth.tar
```
## Acknowledgements
We thank Clement Pinard for his github repository [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch). We use it as our code base. PWCNet is taken from [NVlabs/PWC-Net](https://github.com/NVlabs/PWC-Net). SPyNet implementation is taken from [sniklaus/pytorch-spynet](https://github.com/sniklaus/pytorch-spynet). Correlation module is taken from [ClementPinard/Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension).
