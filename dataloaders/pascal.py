import torch, cv2
import errno
import hashlib
import os
import sys
import tarfile
import numpy as np

import torch.utils.data as data
from PIL import Image
from six.moves import urllib
import json
from mypath import Path


class VOCSegmentation(data.Dataset):

    URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    FILE = "VOCtrainval_11-May-2012.tar"
    MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
    BASE_DIR = 'VOCdevkit/VOC2012'

    category_names = ['background',
                      'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                      'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person',
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self,
                 root=Path.db_root_dir('pascal'),
                 split='val',
                 transform=None,
                 download=False,
                 preprocess=False,
                 area_thres=0,
                 retname=True,
                 suppress_void_pixels=True,
                 default=False):

        self.root = root
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationObject')
        _cat_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        self.transform = transform
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        self.area_thres = area_thres
        self.retname = retname
        self.suppress_void_pixels = suppress_void_pixels
        self.default = default

        # Build the ids file
        area_th_str = ""
        if self.area_thres != 0:
            area_th_str = '_area_thres-' + str(area_thres)

        self.obj_list_file = os.path.join(self.root, self.BASE_DIR, 'ImageSets', 'Segmentation',
                                          '_'.join(self.split) + '_instances' + area_th_str + '.txt')

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.masks = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(_image_dir, line + ".jpg")
                _cat = os.path.join(_cat_dir, line + ".png")
                _mask = os.path.join(_mask_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_mask)
                self.im_ids.append(line.rstrip('\n'))
                self.images.append(_image)
                self.categories.append(_cat)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))
        assert (len(self.images) == len(self.categories))

        # Precompute the list of objects and their categories for each image 
        if (not self._check_preprocess()) or preprocess:
            print('Preprocessing of PASCAL VOC dataset, this will take long, but it will be done only once.')
            self._preprocess()
            
        # Build the list of objects
        self.obj_list = []
        num_images = 0
        for ii in range(len(self.im_ids)):
            flag = False
            for jj in range(len(self.obj_dict[self.im_ids[ii]])):
                if self.obj_dict[self.im_ids[ii]][jj] != -1:
                    self.obj_list.append([ii, jj])
                    flag = True
            if flag:
                num_images += 1

        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(num_images, len(self.obj_list)))

    def __getitem__(self, index):
        _img, _target, _void_pixels, _, _, _ = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target, 'void_pixels': _void_pixels}

        if self.retname:
            _im_ii = self.obj_list[index][0]
            _obj_ii = self.obj_list[index][1]
            sample['meta'] = {'image': str(self.im_ids[_im_ii]),
                              'object': str(_obj_ii),
                              'category': self.obj_dict[self.im_ids[_im_ii]][_obj_ii],
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.obj_list)

    def _check_integrity(self):
        _fpath = os.path.join(self.root, self.FILE)
        if not os.path.isfile(_fpath):
            print("{} does not exist".format(_fpath))
            return False
        _md5c = hashlib.md5(open(_fpath, 'rb').read()).hexdigest()
        if _md5c != self.MD5:
            print(" MD5({}) did not match MD5({}) expected for {}".format(
                _md5c, self.MD5, _fpath))
            return False
        return True

    def _check_preprocess(self):
        _obj_list_file = self.obj_list_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.obj_dict = json.load(open(_obj_list_file, 'r'))

            return list(np.sort([str(x) for x in self.obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess(self):
        self.obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            _mask = np.array(Image.open(self.masks[ii]))
            _mask_ids = np.unique(_mask)
            if _mask_ids[-1] == 255:
                n_obj = _mask_ids[-2]
            else:
                n_obj = _mask_ids[-1]

            # Get the categories from these objects
            _cats = np.array(Image.open(self.categories[ii]))
            _cat_ids = []
            for jj in range(n_obj):
                tmp = np.where(_mask == jj + 1)
                obj_area = len(tmp[0])
                if obj_area > self.area_thres:
                    _cat_ids.append(int(_cats[tmp[0][0], tmp[1][0]]))
                else:
                    _cat_ids.append(-1)
                obj_counter += 1

            self.obj_dict[self.im_ids[ii]] = _cat_ids

        with open(self.obj_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')

    def _download(self):
        _fpath = os.path.join(self.root, self.FILE)

        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        else:
            print('Downloading ' + self.URL + ' to ' + _fpath)

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> %s %.1f%%' %
                                 (_fpath, float(count * block_size) /
                                  float(total_size) * 100.0))
                sys.stdout.flush()

            urllib.request.urlretrieve(self.URL, _fpath, _progress)

        # extract file
        cwd = os.getcwd()
        print('Extracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def _make_img_gt_point_pair(self, index):
        _im_ii = self.obj_list[index][0]
        _obj_ii = self.obj_list[index][1]

        # Read Image
        _img = np.array(Image.open(self.images[_im_ii]).convert('RGB')).astype(np.float32)

        # Read Target object
        _tmp = (np.array(Image.open(self.masks[_im_ii]))).astype(np.float32)
        _void_pixels = (_tmp == 255)
        _tmp[_void_pixels] = 0

        _other_same_class = np.zeros(_tmp.shape)
        _other_classes = np.zeros(_tmp.shape)

        if self.default:
            _target = _tmp
            _background = np.logical_and(_tmp == 0, ~_void_pixels)
        else:
            _target = (_tmp == (_obj_ii + 1)).astype(np.float32)
            _background = np.logical_and(_tmp == 0, ~_void_pixels)
            obj_cat = self.obj_dict[self.im_ids[_im_ii]][_obj_ii]
            for ii in range(1, np.max(_tmp).astype(np.int)+1):
                ii_cat = self.obj_dict[self.im_ids[_im_ii]][ii-1]
                if obj_cat == ii_cat and ii != _obj_ii+1:
                    _other_same_class = np.logical_or(_other_same_class, _tmp == ii)
                elif ii != _obj_ii+1:
                    _other_classes = np.logical_or(_other_classes, _tmp == ii)

        return _img, _target, _void_pixels.astype(np.float32), \
               _other_classes.astype(np.float32), _other_same_class.astype(np.float32), \
               _background.astype(np.float32)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ',area_thres=' + str(self.area_thres) + ')'


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import dataloaders.helpers as helpers
    import torch
    import dataloaders.custom_transforms as tr
    from torchvision import transforms

    transform = transforms.Compose([tr.ToTensor()])

    dataset = VOCSegmentation(split=['train', 'val'], transform=transform, retname=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataloader):
        plt.figure()
        overlay = helpers.overlay_mask(helpers.tens2image(sample["image"]) / 255.,
                                       np.squeeze(helpers.tens2image(sample["gt"])))
        plt.imshow(overlay)
        plt.title(dataset.category_names[sample["meta"]["category"][0]])
        if i == 3:
            break

    plt.show(block=True)
