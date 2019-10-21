# Pytorch implementation of SiamFC
## Introduction
This project is the Pytorch implementation of Fully-Convolutional Siamese Networks for Object Tracking, the original version was trainned on ILSVRC2015-VID dataset, but this version is trained on GOT-10K dataset, and achieved better results.
## Download models
Download models in [BaiduYun](https://pan.baidu.com/s/1pBZob53r8On-eJBKfY-qKQ&shfl=sharepset) and put the model.pth in the correct directory in experiments.
The extracted code is ```bash duuy```
## Run demo
```bash
cd SiamFC-Pytorch

mkdir models

# for color model
wget http://www.robots.ox.ac.uk/%7Eluca/stuff/siam-fc_nets/2016-08-17.net.mat -P models/
# for color+gray model
wget http://www.robots.ox.ac.uk/%7Eluca/stuff/siam-fc_nets/2016-08-17_gray025.net.mat -P models/

python bin/convert_pretrained_model.py

# video dir should conatin groundtruth_rect.txt which the same format like otb
python bin/demo_siamfc --gpu-id [gpu_id] --video-dir path/to/video
```

## Training
Download ILSVRC2015-VID 

```bash
cd SiamFC-Pytorch

mkdir models

# using 12 threads should take an hour
python bin/create_dataset.py --data-dir path/to/data/ILSVRC2015 \
			     --output-dir path/to/data/ILSVRC_VID_CURATION \
			     --num-threads 8

# ILSVRC2015_VID_CURATION and ILSVRC2015_VID_CURATION.lmdb should be in the same directory
# the ILSVRC2015_VID_CURATION.lmdb should be about 34G or so
python bin/create_lmdb.py --data-dir path/to/data/ILSVRC_VID_CURATION \
			  --output-dir path/to/data/ILSVRC2015_VID_CURATION.lmdb \
		          --num-threads 8

# training should take about 1.5~2hrs on a Titan Xp GPU with 30 epochs
python bin/train_siamfc.py --gpu-id [gpu_id] --data-dir path/to/data/ILSVRC2015_VID_CURATION
```
## Benchmark results
#### OTB100

| Tracker 			    		 |  AUC   |
| ---------------------------------------------  | -------|
| SiamFC-color(converted from matconvnet)        | 0.5544 |
| SiamFC-color+gray(converted from matconvnet)   | 0.5818 |
| SiamFC(trained from scratch)      		 | 0.6230 |

## Reference
[1] Bertinetto, Luca and Valmadre, Jack and Henriques, Joo F and Vedaldi, Andrea and Torr, Philip H S
		Fully-Convolutional Siamese Networks for Object Tracking
		In ECCV 2016 workshops
[2] [StrangerZhang/SiamFC-PyTorch](https://github.com/StrangerZhang/SiamFC-PyTorch)
