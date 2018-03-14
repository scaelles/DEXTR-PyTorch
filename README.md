# Deep Extreme Cut (DEXTR)
Visit our [project page](http://www.vision.ee.ethz.ch/~cvlsegmentation) for accessing the paper, and the pre-computed results.

![DEXTR](doc/osvos.png)

This is the implementation of our work `Deep Extreme Cut (DEXTR)`, for object segmentation from extreme points.

### Installation
The code was tested with Miniconda and Python 3.6. After installing the Miniconda environment:
0. Clone the repo:
  ```Shell
  git clone https://github.com/scaelles/DEXTR-PyTorch
  cd DEXTR-PyTorch
  ```
 
1. Install dependencies:
  ```Shell
  conda install pytorch torchvision -c pytorch
  conda install matplotlib opencv-python pillow pillow scikit-learn scikit-image
  ```
  
2. Download the models by running the script inside ```models/```:
  ```Shell
  cd models/
  chmod +x download_dextr_models.sh
  ./download_dextr_models.sh
  cd ..
  ```

3. To try the demo version of DEXTR, please run:
  ```Shell
  python demo.py
  ```

To train and evaluate DEXTR on PASCAL (or PASCAL + SBD), please follow these additional steps:

4. Install tensorboard (integrated with PyTorch). 
  ```Shell
  pip install tensorboard tensorboardx
  ```

5. Download the pre-trained PSPNet model for semantic segmentation, taken from [this](https://github.com/isht7/pytorch-deeplab-resnet) repository.
  ```Shell
  cd models/
  chmod +x download_pretrained_psp_model.sh
  ./download_pretrained_psp_models.sh
  cd ..
  ```
6. Run ```train_pascal.py```, after changing the default parameters, if necessary (eg. gpu_id).

Enjoy!!


If you use this code, please consider citing the following paper:

	@Inproceedings{Man+18,
	  Title          = { Deep Extreme Cut: From Extreme Points to Object Segmentation},
	  Author         = {K.K. Maninis and S. Caelles and J. Pont-Tuset and L. {Van Gool}},
	  Booktitle      = {Computer Vision and Pattern Recognition (CVPR)},
	  Year           = {2018}
	}

We thank the authors of [pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet) for making their PyTorch re-implementation of DeepLab-v2 available!

If you encounter any problem or/and you have any suggestions please contact us at {kmaninis, scaelles}@vision.ee.ethz.ch
