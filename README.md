# DEXTR

ToDo list:
- Verify that the pascal training gets ~89.5 (Currently training)
- Check ToDO in the code and commit

Installation
Install Miniconda for python 3.6
conda install pytorch torchvision -c pytorch
conda install matplotlib opencv tensorflow tensorflow-tensorboard pillow

pip install tensorboardX

Set up
Place the weights in models:
Pascal:https://data.vision.ee.ethz.ch/csergi/share/DEXTR/MS_DeepLab_resnet_trained_VOC.pth
COCO:https://data.vision.ee.ethz.ch/csergi/share/DEXTR/MS_DeepLab_resnet_pretrained_COCO_init.pth
DEXTR:https://data.vision.ee.ethz.ch/csergi/share/DEXTR/dextr_pascal-sbd_epoch-99.pth

Thank also other repos that we used weights and code (DeepLab Pytorch)
