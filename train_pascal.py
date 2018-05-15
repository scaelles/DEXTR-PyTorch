import socket
import timeit
from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
import glob

# PyTorch includes
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import upsample

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders.combine_dbs import CombineDBs as combine_dbs
import dataloaders.pascal as pascal
import dataloaders.sbd as sbd
from dataloaders import custom_transforms as tr
import networks.deeplab_resnet as resnet
from layers.loss import class_balanced_cross_entropy_loss
from dataloaders.helpers import *

# Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

# Setting parameters
use_sbd = False
nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume

p = OrderedDict()  # Parameters to include in report
classifier = 'psp'  # Head classifier to use
p['trainBatch'] = 5  # Training batch size
testBatch = 5  # Testing batch size
useTest = 1  # See evolution of the test set when training?
nTestInterval = 10  # Run on test set every nTestInterval epochs
snapshot = 20  # Store a model every snapshot epochs
relax_crop = 50  # Enlarge the bounding box by relax_crop pixels
nInputChannels = 4  # Number of input channels (RGB + heatmap of extreme points)
zero_pad_crop = True  # Insert zero padding when cropping the image
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-8  # Learning rate
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.9  # Momentum

# Results and model directories (a new directory is generated for every run)
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
if resume_epoch == 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
else:
    run_id = 0
save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
if not os.path.exists(os.path.join(save_dir, 'models')):
    os.makedirs(os.path.join(save_dir, 'models'))

# Network definition
modelName = 'dextr_pascal'
net = resnet.resnet101(1, pretrained=True, nInputChannels=nInputChannels, classifier=classifier)
if resume_epoch == 0:
    print("Initializing from pretrained Deeplab-v2 model")
else:
    print("Initializing weights from: {}".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage))
train_params = [{'params': resnet.get_1x_lr_params(net), 'lr': p['lr']},
                {'params': resnet.get_10x_lr_params(net), 'lr': p['lr'] * 10}]

net.to(device)

# Training the network
if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    # Preparation of the data loaders
    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
        tr.CropFromMask(crop_elems=('image', 'gt'), relax=relax_crop, zero_pad=zero_pad_crop),
        tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)}),
        tr.ExtremePoints(sigma=10, pert=5, elem='crop_gt'),
        tr.ToImage(norm_elem='extreme_points'),
        tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
        tr.ToTensor()])
    composed_transforms_ts = transforms.Compose([
        tr.CropFromMask(crop_elems=('image', 'gt'), relax=relax_crop, zero_pad=zero_pad_crop),
        tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)}),
        tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
        tr.ToImage(norm_elem='extreme_points'),
        tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
        tr.ToTensor()])

    voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)
    voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)

    if use_sbd:
        sbd = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr, retname=True)
        db_train = combine_dbs([voc_train, sbd], excluded=[voc_val])
    else:
        db_train = voc_train

    p['dataset_train'] = str(db_train)
    p['transformations_train'] = [str(tran) for tran in composed_transforms_tr.transforms]
    p['dataset_test'] = str(db_train)
    p['transformations_test'] = [str(tran) for tran in composed_transforms_ts.transforms]

    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2)
    testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=2)

    generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    # Train variables
    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    print("Training Network")
    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()

        net.train()
        for ii, sample_batched in enumerate(trainloader):

            inputs, gts = sample_batched['concat'], sample_batched['crop_gt']

            # Forward-Backward of the mini-batch
            inputs.requires_grad_()
            inputs, gts = inputs.to(device), gts.to(device)

            output = net.forward(inputs)
            output = upsample(output, size=(512, 512), mode='bilinear', align_corners=True)

            # Compute the losses, side outputs and fuse
            loss = class_balanced_cross_entropy_loss(output, gts, size_average=False, batch_average=True)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == num_img_tr - 1:
                running_loss_tr = running_loss_tr / num_img_tr
                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii*p['trainBatch']+inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time)+"\n")

            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

        # Save the model
        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            net.eval()
            with torch.no_grad():
                for ii, sample_batched in enumerate(testloader):
                    inputs, gts = sample_batched['concat'], sample_batched['crop_gt']

                    # Forward pass of the mini-batch
                    inputs, gts = inputs.to(device), gts.to(device)

                    output = net.forward(inputs)
                    output = upsample(output, size=(512, 512), mode='bilinear', align_corners=True)

                    # Compute the losses, side outputs and fuse
                    loss = class_balanced_cross_entropy_loss(output, gts, size_average=False)
                    running_loss_ts += loss.item()

                    # Print stuff
                    if ii % num_img_ts == num_img_ts - 1:
                        running_loss_ts = running_loss_ts / num_img_ts
                        print('[Epoch: %d, numImages: %5d]' % (epoch, ii*testBatch+inputs.data.shape[0]))
                        writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
                        print('Loss: %f' % running_loss_ts)
                        running_loss_ts = 0

    writer.close()

# Generate result of the validation images
net.eval()
composed_transforms_ts = transforms.Compose([
    tr.CropFromMask(crop_elems=('image', 'gt'), relax=relax_crop, zero_pad=zero_pad_crop),
    tr.FixedResize(resolutions={'gt': None, 'crop_image': (512, 512), 'crop_gt': (512, 512)}),
    tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
    tr.ToImage(norm_elem='extreme_points'),
    tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
    tr.ToTensor()])
db_test = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts, retname=True)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

save_dir_res = os.path.join(save_dir, 'Results')
if not os.path.exists(save_dir_res):
    os.makedirs(save_dir_res)

print('Testing Network')
with torch.no_grad():
    # Main Testing Loop
    for ii, sample_batched in enumerate(testloader):

        inputs, gts, metas = sample_batched['concat'], sample_batched['gt'], sample_batched['meta']

        # Forward of the mini-batch
        inputs = inputs.to(device)

        outputs = net.forward(inputs)
        outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
        outputs = outputs.to(torch.device('cpu'))

        for jj in range(int(inputs.size()[0])):
            pred = np.transpose(outputs.data.numpy()[jj, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)
            gt = tens2image(gts[jj, :, :, :])
            bbox = get_bbox(gt, pad=relax_crop, zero_pad=zero_pad_crop)
            result = crop2fullmask(pred, bbox, gt, zero_pad=zero_pad_crop, relax=relax_crop)

            # Save the result, attention to the index jj
            sm.imsave(os.path.join(save_dir_res, metas['image'][jj] + '-' + metas['object'][jj] + '.png'), result)
