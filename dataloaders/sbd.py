from __future__ import print_function, division

import torch, cv2
import errno
import hashlib
import json
import os
import sys
import tarfile

import numpy as np
import scipy.io
import torch.utils.data as data
from PIL import Image
from six.moves import urllib
from mypath import Path


class SBDSegmentation(data.Dataset):

    URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
    FILE = "benchmark.tgz"
    MD5 = '82b4d87ceb2ed10f6038a1cba92111cb'

    def __init__(self,
                 root=Path.db_root_dir('sbd'),
                 split='val',
                 transform=None,
                 download=False,
                 preprocess=False,
                 area_thres=0,
                 retname=True):

        # Store parameters
        self.root = root
        self.transform = transform
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        self.area_thres = area_thres
        self.retname = retname

        # Where to find things according to the author's structure
        self.dataset_dir = os.path.join(self.root, 'benchmark_RELEASE', 'dataset')
        _mask_dir = os.path.join(self.dataset_dir, 'inst')
        _image_dir = os.path.join(self.dataset_dir, 'img')

        if self.area_thres != 0:
            self.obj_list_file = os.path.join(self.dataset_dir, '_'.join(self.split) + '_instances_area_thres-' +
                                              str(area_thres) + '.txt')
        else:
            self.obj_list_file = os.path.join(self.dataset_dir, '_'.join(self.split) + '_instances' + '.txt')

        # Download dataset?
        if download:
            self._download()
            if not self._check_integrity():
                raise RuntimeError('Dataset file downloaded is corrupted.')

        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.images = []
        self.masks = []
        for splt in self.split:
            with open(os.path.join(self.dataset_dir, splt+'.txt'), "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                _image = os.path.join(_image_dir, line + ".jpg")
                _mask = os.path.join(_mask_dir, line + ".mat")
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.im_ids.append(line)
                self.images.append(_image)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

        # Precompute the list of objects and their categories for each image 
        if (not self._check_preprocess()) or preprocess:
            print('Preprocessing SBD dataset, this will take long, but it will be done only once.')
            self._preprocess()
            
        # Build the list of objects
        self.obj_list = []
        num_images = 0
        for ii in range(len(self.im_ids)):
            if self.im_ids[ii] in self.obj_dict.keys():
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

        _img, _target = self._make_img_gt_point_pair(index)
        _void_pixels = (_target == 255).astype(np.float32)
        sample = {'image': _img, 'gt': _target, 'void_pixels': _void_pixels}

        if self.retname:
            _im_ii = self.obj_list[index][0]
            _obj_ii = self.obj_list[index][1]
            sample['meta'] = {'image': str(self.im_ids[_im_ii]),
                              'object': str(_obj_ii),
                              'im_size': (_img.shape[0], _img.shape[1]),
                              'category': self.obj_dict[self.im_ids[_im_ii]][_obj_ii]}

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
        # Check that the file with categories is there and with correct size
        _obj_list_file = self.obj_list_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.obj_dict = json.load(open(_obj_list_file, 'r'))
            return list(np.sort([str(x) for x in self.obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess(self):
        # Get all object instances and their category
        self.obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            tmp = scipy.io.loadmat(self.masks[ii])
            _mask = tmp["GTinst"][0]["Segmentation"][0]
            _cat_ids = tmp["GTinst"][0]["Categories"][0].astype(int)

            _mask_ids = np.unique(_mask)
            n_obj = _mask_ids[-1]
            assert(n_obj == len(_cat_ids))

            for jj in range(n_obj):
                temp = np.where(_mask == jj + 1)
                obj_area = len(temp[0])
                if obj_area < self.area_thres:
                    _cat_ids[jj] = -1
                obj_counter += 1

            self.obj_dict[self.im_ids[ii]] = np.squeeze(_cat_ids, 1).tolist()

        # Save it to file for future reference
        with open(self.obj_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Pre-processing finished')

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

        # Read Taret object
        _tmp = scipy.io.loadmat(self.masks[_im_ii])["GTinst"][0]["Segmentation"][0]
        _target = (_tmp == (_obj_ii + 1)).astype(np.float32)

        return _img, _target

    def __str__(self):
        return 'SBDSegmentation(split='+str(self.split)+', area_thres='+str(self.area_thres)+')'


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import dataloaders.helpers as helpers
    import torch
    import torchvision.transforms as transforms
    import dataloaders.custom_transforms as tr

    transform = transforms.Compose([tr.ToTensor()])
    dataset = SBDSegmentation(transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    for i, data in enumerate(dataloader):
        plt.figure()
        plt.imshow(helpers.tens2image(data['image'])/255)
        plt.figure()
        plt.imshow(helpers.tens2image(data['gt'])[:, :, 0])
        if i == 10:
            break

    plt.show(block=True)
