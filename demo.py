# ToDo Remove before release
from PyQt5 import QtCore, QtGui, QtWidgets

import os

import torch, cv2

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


from torch.nn.functional import upsample
from torch.autograd import Variable

import networks.deeplab_resnet as resnet
from mypath import Path
from dataloaders import helpers as helpers

modelName = 'dextr_pascal-sbd'
epoch = 100
pad = 50
thres = 0.9

#  Read image and click the points
image = np.array(Image.open('demo.jpg'))
plt.figure()
plt.imshow(image)
plt.title('Click the four extreme points')
extreme_points_ori = np.array(plt.ginput(4)).astype(np.int)
plt.close()

#  Create the network and load the weights
net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
print("Initializing weights from: {}".format(
    os.path.join(Path.models_dir(), modelName + '_epoch-' + str(epoch - 1) + '.pth')))
net.load_state_dict(
    torch.load(os.path.join(Path.models_dir(), modelName + '_epoch-' + str(epoch - 1) + '.pth'),
               map_location=lambda storage, loc: storage))
net.eval()

#  Crop image to the bounding box from the extreme points and resize
bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=pad, zero_pad=True)
crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

#  Generate extreme point heat map normalized to image values
extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [pad, pad]
extreme_points = (512*extreme_points*[1/crop_image.shape[1], 1/crop_image.shape[0]]).astype(np.int)
extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

#  Concatenate inputs and convert to tensor
input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
input_dextr = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

inputs = Variable(input_dextr, volatile=True)
outputs = net.forward(inputs)
outputs = upsample(outputs, size=(512, 512), mode='bilinear')
pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
pred = 1 / (1 + np.exp(-pred))
pred = np.squeeze(pred)
result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres

plt.imshow(helpers.overlay_mask(image/255, result))
plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')
plt.show()
