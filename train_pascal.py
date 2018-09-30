import socket
import timeit
from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
import argparse

# PyTorch includes
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders.combine_dbs import CombineDBs as combine_dbs
import dataloaders.pascal as pascal
import dataloaders.sbd as SBD
from dataloaders import custom_transforms as tr
import networks.deeplab_resnet as resnet
from layers.loss import BalancedCrossEntropyLoss
from dataloaders.helpers import *
from evaluation.eval import eval_one_result  # TODO: Interface
from mypath import Path


def parse_args():
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser(description='DEXTR-Experiments')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Which GPU to be used')
    parser.add_argument('--use_sbd', type=str2bool, default=False,
                        help='Use SBD dataset in addition to PASCAL VOC?')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Resume from checkpoint')
    parser.add_argument('--classifier', type=str, default='psp', choices=['psp', 'atrous'],
                        help='Classifier head to use')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=1e-8,
                        help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight Decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')

    parser.add_argument('--train_norm', type=str2bool, default=False,
                        help='train batch norm parameters?')

    parser.add_argument('--use_test', type=str2bool, default=True,
                        help='Use testing during training?')
    parser.add_argument('--test_interval', type=int, default=10,
                        help='Test every test_interval epochs of training')
    parser.add_argument('--snapshot', type=int, default=50,
                        help='Take a snapshot of the model every snapshot iterations')
    parser.add_argument('--relax_crop', type=int, default=50,
                        help='Relax tight bbox around the object by relax_crop pixels on each side')
    parser.add_argument('--zero_pad_crop', type=str2bool, default=True,
                        help='Insert zero padding when cropping an image')
    parser.add_argument('--n_ave_grad', type=int, default=1,
                        help='Average the gradients over n_ave_grad forward passes, useful when batch size is small')

    parser.add_argument('--evaluate', type=str2bool, default=True,
                        help='Evaluate Results in the end of training')

    return parser.parse_args()


def main():
    args = parse_args()

    p = OrderedDict()  # Parameters to include in report
    p['use_sbd'] = args.use_sbd
    p['classifier'] = args.classifier
    p['batch_size'] = args.batch_size
    p['n_epochs'] = args.n_epochs
    p['learning_rate'] = args.learning_rate
    p['weight_decay'] = args.weight_decay
    p['momentum'] = args.momentum
    p['train_norm'] = args.train_norm

    # Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using GPU: {} '.format(args.gpu_id))

    # Results and model directories (a new directory is generated for every run)
    exp_name = 'use_sbd-' + str(args.use_sbd) + \
               '_train_norm-' + str(args.train_norm) + \
               '_batch_size-' + str(args.batch_size)

    save_dir = os.path.join(Path.save_dir_root(), exp_name)
    if not os.path.exists(os.path.join(save_dir, 'models')):
        os.makedirs(os.path.join(save_dir, 'models'))

    # Network definition
    model_name = 'dextr_pascal'
    net = resnet.resnet101(1, pretrained=True,
                           n_input_channels=4,
                           classifier=args.classifier,
                           train_norm=args.train_norm)

    if args.resume_epoch == 0:
        print("Initializing from pretrained Deeplab-v2 model")
    else:
        print("Initializing weights from: {}".format(
            os.path.join(save_dir, 'models', model_name + '_epoch-' + str(args.resume_epoch - 1) + '.pth')))
        net.load_state_dict(
            torch.load(os.path.join(save_dir, 'models', model_name + '_epoch-' + str(args.resume_epoch - 1) + '.pth'),
                       map_location=lambda storage, loc: storage))

    net.to(device)

    # Training the network
    if args.resume_epoch != args.n_epochs:
        # Logging into Tensorboard
        log_dir = os.path.join(save_dir, 'models',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)

        # Use the following optimizer
        train_params = [{'params': resnet.get_1x_lr_params(net), 'lr': args.learning_rate},
                        {'params': resnet.get_10x_lr_params(net), 'lr': args.learning_rate * 10}]
        optimizer = optim.SGD(train_params,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
        p['optimizer'] = str(optimizer)

        # Preparation of the data loaders
        composed_transforms_tr = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=args.relax_crop, zero_pad=args.zero_pad_crop),
            tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)}),
            tr.ExtremePoints(sigma=10, pert=5, elem='crop_gt'),
            tr.ToImage(norm_elem='extreme_points'),
            tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
            tr.ToTensor()])
        composed_transforms_ts = transforms.Compose([
            tr.CropFromMask(crop_elems=('image', 'gt'), relax=args.relax_crop, zero_pad=args.zero_pad_crop),
            tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)}),
            tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
            tr.ToImage(norm_elem='extreme_points'),
            tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
            tr.ToTensor()])

        voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)
        voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)

        if args.use_sbd:
            sbd = SBD.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr, retname=True)
            db_train = combine_dbs([voc_train, sbd], excluded=[voc_val])
        else:
            db_train = voc_train

        p['dataset_train'] = str(db_train)
        p['transformations_train'] = [str(tran) for tran in composed_transforms_tr.transforms]
        p['dataset_test'] = str(db_train)
        p['transformations_test'] = [str(tran) for tran in composed_transforms_ts.transforms]

        trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = DataLoader(voc_val, batch_size=args.batch_size, shuffle=False, num_workers=2)

        criterion = BalancedCrossEntropyLoss(size_average=False, batch_average=True)

        generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

        # Train variables
        num_img_tr = len(trainloader)
        num_img_ts = len(testloader)
        running_loss_tr = 0.0
        running_loss_ts = 0.0
        ave_grad = 0
        print("Training Network")
        # Main Training and Testing Loop
        for epoch in range(args.resume_epoch, args.n_epochs):
            start_time = timeit.default_timer()

            if args.train_norm:
                net.train()
            else:
                net.eval()

            for ii, sample_batched in enumerate(trainloader):

                inputs, gts = sample_batched['concat'], sample_batched['crop_gt']

                # Forward-Backward of the mini-batch
                inputs.requires_grad_()
                inputs, gts = inputs.to(device), gts.to(device)

                output = net.forward(inputs)
                output = interpolate(output, size=(512, 512), mode='bilinear', align_corners=True)

                # Compute the losses, side outputs and fuse
                loss = criterion(output, gts)
                running_loss_tr += loss.item()

                # Print stuff
                if ii % num_img_tr == num_img_tr - 1:
                    running_loss_tr = running_loss_tr / num_img_tr
                    writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * args.batch_size + inputs.data.shape[0]))
                    print('Loss: %f' % running_loss_tr)
                    running_loss_tr = 0
                    stop_time = timeit.default_timer()
                    print("Execution time: " + str(stop_time - start_time) + "\n")

                # Backward the averaged gradient
                loss /= args.n_ave_grad
                loss.backward()
                ave_grad += 1

                # Update the weights once in n_ave_grad forward passes
                if ave_grad % args.n_ave_grad == 0:
                    writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                    optimizer.step()
                    optimizer.zero_grad()
                    ave_grad = 0

            # Save the model
            if (epoch % args.snapshot) == args.snapshot - 1 and epoch != 0:
                torch.save(net.state_dict(), os.path.join(save_dir, 'models',
                                                          model_name + '_epoch-' + str(epoch) + '.pth'))

            # One testing epoch
            if args.use_test and epoch % args.test_interval == (args.test_interval - 1):
                net.eval()
                with torch.no_grad():
                    for ii, sample_batched in enumerate(testloader):
                        inputs, gts = sample_batched['concat'], sample_batched['crop_gt']

                        # Forward pass of the mini-batch
                        inputs, gts = inputs.to(device), gts.to(device)

                        output = net.forward(inputs)
                        output = interpolate(output, size=(512, 512), mode='bilinear', align_corners=True)

                        # Compute the losses, side outputs and fuse
                        loss = criterion(output, gts)
                        running_loss_ts += loss.item()

                        # Print stuff
                        if ii % num_img_ts == num_img_ts - 1:
                            running_loss_ts = running_loss_ts / num_img_ts
                            print('[Epoch: %d, numImages: %5d]' % (epoch, ii * args.batch_size + inputs.data.shape[0]))
                            writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
                            print('Loss: %f' % running_loss_ts)
                            running_loss_ts = 0

        writer.close()

    # Generate result of the validation images
    net.eval()
    composed_transforms_ts = transforms.Compose([
        tr.CropFromMask(crop_elems=('image', 'gt'), relax=args.relax_crop, zero_pad=args.zero_pad_crop),
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
            outputs = interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=True)
            outputs = outputs.to(torch.device('cpu'))

            for jj in range(int(inputs.size()[0])):
                pred = np.transpose(outputs.data.numpy()[jj, :, :, :], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = np.squeeze(pred)
                gt = tens2image(gts[jj, :, :, :])
                bbox = get_bbox(gt, pad=args.relax_crop, zero_pad=args.zero_pad_crop)
                result = crop2fullmask(pred, bbox, gt, zero_pad=args.zero_pad_crop, relax=args.relax_crop)

                # Save the result, attention to the index jj
                sm.imsave(os.path.join(save_dir_res, metas['image'][jj] + '-' + metas['object'][jj] + '.png'), result)


if __name__ == '__main__':
    main()
