# Pytorch implementation of SiamFC
## Introduction
This project is the Pytorch implementation of [**Fully-Convolutional Siamese Networks for Object Tracking**](http://www.robots.ox.ac.uk/~luca/siamese-fc.html), the original version was trainned on **ILSVRC2015-VID** dataset, but this version was trained on [**GOT-10K**](http://got-10k.aitestunion.com/) dataset, and achieved better results.
## Download models
Download models in [BaiduYun](https://pan.baidu.com/s/1pBZob53r8On-eJBKfY-qKQ&shfl=sharepset) and put the model.pth in the correct directory in experiments.
The extracted code is ```duuy```
## Run tracker
```bash
cd siamfc-pytorch
python run_tracker.py --data_dir path/to/data --model_dir path/to/model
```
## Training
Download GOT-10K

```bash
cd siamfc-pytorch
# data preprocessing
python data_preprocessing.py --data-dir path/to/data/GOT-10K \
			     --output-dir path/to/data/GOT-10K/crop_data \
			     --num_processings 8
# training 
python train.py --train_data_dir path/to/data/GOT-10K/crop_train_data  \
			     --val_data_dir path/to/data/GOT-10K/crop_val_data 
```
## Benchmark results
#### OTB100

| Tracker 			    		 |  AUC   |
| ---------------------------------------------  | -------|
| SiamFC-color(converted from matconvnet)        | 0.5544 |
| SiamFC-color+gray(converted from matconvnet)   | 0.5818 |
| SiamFC(trained from scratch)      		 | **0.6230** |

## Reference
[1] Bertinetto, Luca and Valmadre, Jack and Henriques, Joo F and Vedaldi, Andrea and Torr, Philip H S
		Fully-Convolutional Siamese Networks for Object Tracking
		In ECCV 2016 workshops                                       		   
[2] [StrangerZhang/SiamFC-PyTorch](https://github.com/StrangerZhang/SiamFC-PyTorch)
