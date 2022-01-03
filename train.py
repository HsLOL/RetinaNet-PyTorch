import argparse
from config import cfg
import torch
import numpy as np
from datasets.SSDD_dataset import SSDDataset
from datasets.collater import Collater
import torch.utils.data as data
from utils.utils import set_random_seed, count_param
from models.model import RetinaNet
import torch.optim as optim
from tqdm import tqdm
import os
from warmup import WarmupLR
import glob
from tensorboardX import SummaryWriter
import datetime
import torch.nn as nn
from eval import evaluate

# supported evaluation methods: [coco, voc]
metrics = ['coco', 'voc']


def get_args():
    parser = argparse.ArgumentParser()

    # network
    parser.add_argument('--backbone', type=str, default='resnet50')

    # training set
    parser.add_argument('--dataset', type=str, default=r'SSDD')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The number of images per batch among all devices')
    parser.add_argument('--multi-scale', action='store_true',
                        help='adjust (67% - 150%) image size')  # Not Supported yet
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--freeze_bn', action='store_false', help='freeze backbone BN parameters.')
    parser.add_argument('--warm_up', default=cfg.warmup_action, help='launch warmup.')
    parser.add_argument('--eval_method', type=str, default='coco', help='the evaluation metric.')

    # parser.add_argument('--last', type=str,
    #                     # default=r'checkpoints/last/last.pth',
    #                     default=None,  # set None will train model from scratch
    #                     help='load training from last epoch.'
    #                          'if set None model will train from scratch '
    #                          'else model will continue from last epoch.')
    parser.add_argument('--load', type=str, default=None, help='load training from best.pth,'
                                                               'while begin from 0 epoch.'
                                                               'set None, model will train from scratch.')
    parser.add_argument('--resume', type=str, default=None,
                        help='load training from the stop epoch or steps.'
                             'for example:'
                             'default could be {epoch}_{step}.pth, set None model will train from scratch.')

    # model output
    parser.add_argument('--log_path', type=str, default=r'logs', help='The output result.txt relative path.')
    parser.add_argument('--tensorboard', type=str, default=r'tensorboard',
                        help='The output tensorboard file relative path')
    parser.add_argument('--checkpoint', type=str, default=r'checkpoints', help='the output weight file relative path.')

    # common setting
    parser.add_argument('--device', type=int, default=0, help='The index of the GPUs')
    args = parser.parse_args()
    return args


def train(args):
    epochs = args.epoch
    device = args.device if torch.cuda.is_available() else "cpu"
    results_file = 'result.txt'  # output log file

    weight = ''
    if args.load:
        weight = cfg.output_path + os.sep + args.checkpoint + os.sep + 'best' + os.sep + 'best.pth'

    if args.resume:
        weight = cfg.output_path + os.sep + args.checkpoint + os.sep + args.resume.split('/')[-1]

    start_epoch = 0
    best_fitness = 0
    fitness = 0

    # create folder
    tensorboard_path = os.path.join(cfg.output_path, args.tensorboard)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    checkpoint_path = os.path.join(cfg.output_path, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    last_checkpoint_path = os.path.join(checkpoint_path, 'last')
    if not os.path.exists(last_checkpoint_path):
        os.makedirs(last_checkpoint_path)

    best_checkpoint_path = os.path.join(checkpoint_path, 'best')
    if not os.path.exists(best_checkpoint_path):
        os.makedirs(best_checkpoint_path)

    log_path = os.path.join(cfg.output_path, args.log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # for f in glob.glob(log_path):
    #     if os.path.isfile(f):
    #         os.remove(f)  # remove all the log files

    # multi-scale
    if args.multi_scale:
        scales = cfg.image_size + 32 * np.array([x for x in range(-1, 5)])
        # also can set scales manually
        # scales = np.array([384, 480, 544, 608, 704, 800, 896, 960])
    else:
        scales = cfg.image_size

    # dataloader
    # transform means using image argument
    # True: use False: not use
    train_dataset = SSDDataset(root_dir=cfg.data_path, set_name='train', transform=False)
    collater = Collater(scales=scales, keep_ratio=cfg.keep_ratio, multiple=32)
    train_generator = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=4,  # 4 * number of the GPU
        collate_fn=collater,
        shuffle=True,
        pin_memory=True,
        drop_last=True)

    # Initialize model & set random seed
    set_random_seed(seed=42, deterministic=False)
    model = RetinaNet(backbone=args.backbone, loss_func=cfg.loss_func, pretrained=True)
    count_param(model)

    # init tensorboardX
    writer = SummaryWriter(tensorboard_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # Optimizer Option
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    # optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=0.0001)

    # Scheduler Option
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.6, 0.8]], gamma=0.1)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)

    # Warm-up
    is_warmup = False
    if args.warm_up and not args.resume and not args.load:
        print('[Info]: Launching Warmup.')
        scheduler = WarmupLR(scheduler, init_lr=cfg.warmup_lr, num_warmup=cfg.warmup_epoch, warmup_strategy='cos')
        is_warmup = True
    if is_warmup is False:
        print('[Info]: Not Launching Warmup.')
    # scheduler.last_epoch value of -1 indicates that model will start from the begining
    # scheduler.last_epoch = start_epoch - 1

    # Load chkpt
    if args.resume:
        if weight.endswith('.pth'):
            chkpt = torch.load(weight)
            last_step = chkpt['step']
            # Load model
            if 'model' in chkpt.keys():
                model.load_state_dict(chkpt['model'])
            else:
                model.load_state_dict(chkpt)

            # Load optimizer
            if 'optimizer' in chkpt.keys() and chkpt['optimizer'] is not None and args.resume:
                optimizer.load_state_dict(chkpt['optimizer'])
                best_fitness = chkpt['best_fitness']
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):  # from CPU Tensor to GPU Tensor
                            state[k] = v.cuda(device=args.device)

            # Load scheduler
            if 'scheduler' in chkpt.keys() and chkpt['scheduler'] is not None and args.resume:
                scheduler_state = chkpt['scheduler']
                scheduler._step_count = scheduler_state['step_count']
                scheduler.last_epoch  = scheduler_state['last_epoch']

            # Load results
            if 'training_results' in chkpt.keys() and chkpt.get('training_results') is not None and args.resume:
                with open(results_file, 'w') as f_in:
                    f_in.write(chkpt['training_results'])

            if args.resume and 'epoch' in chkpt.keys():
                start_epoch = chkpt['epoch'] + 1

            del chkpt
    elif args.load:
        """ If args.load is True, 
            model will train from specify epoch and specify optimizer and scheduler.
            """
        if weight.endswith('.pth'):
            chkpt = torch.load(weight)
            # Load model
            if 'model' in chkpt.keys():
                model.load_state_dict(chkpt['model'])
            else:
                model.load_state_dict(chkpt)
            last_step = 0
    else:
        last_step = 0

    if torch.cuda.is_available():
        model = model.cuda(device=device)

    # start training
    step = max(0, last_step)
    num_iter_per_epoch = len(train_generator)
    # results = ('P', 'R', 'mAP', 'F1')
    train_results = (0, 0, 0, 0)
    val_results = (0, 0, 0, 0)
    print(('\n' + '%10s' * 8) % ('Epoch', 'Steps', 'gpu_mem', 'cls', 'reg', 'total', 'targets', 'img_size'))
    if is_warmup:
        scheduler.step()
    for epoch in range(start_epoch, epochs):
        last_epoch = step // num_iter_per_epoch
        if epoch < last_epoch:
            continue
        pbar = tqdm(enumerate(train_generator), total=len(train_generator))  # progress bar

        # for each epoch, we set model.eval() to model.train()
        # and freeze backbone BN Layers parameters
        model.train()

        if args.freeze_bn:
            model.freeze_bn()

        for iter, (ni, batch) in enumerate(pbar):

            if iter < step - last_epoch * num_iter_per_epoch:
                pbar.update()
                continue

            optimizer.zero_grad()
            images, annots, image_names = batch['image'], batch['annot'], batch['image_name']
            if torch.cuda.is_available():
                images, annots = images.cuda(device=device), annots.cuda(device=device)
            loss_cls, loss_reg = model(images, annots, image_names)

            # Using .mean() is following Ming71 and Zylo117 repo
            loss_cls = loss_cls.mean()
            loss_reg = loss_reg.mean()

            total_loss = loss_cls + loss_reg

            if not torch.isfinite(total_loss):
                print('[Warning]: loss is nan')
                break

            if bool(total_loss == 0):
                continue

            total_loss.backward()

            # Update parameters

            # if loss is not nan not using grad clip
            # nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            # print batch result
            mem = torch.cuda.memory_reserved(device=device) / 1E9 if torch.cuda.is_available() else 0

            s = ('%10s' * 3 + '%10.3g' * 4 + '%10s' * 1) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g' % iter,
                '%.3gG' % mem, loss_cls.item(), loss_reg.item(), total_loss.item(), annots.shape[1],
                '%gx%g' % (int(images.shape[2]), int(images.shape[3])))
            pbar.set_description(s)

            # write loss info into tensorboard
            writer.add_scalars('Loss', {'train': total_loss}, step)
            writer.add_scalars('Regression_loss', {'train': loss_reg}, step)
            writer.add_scalars('Classfication_loss', {'train': loss_cls}, step)

            # write lr info into tensorboard
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr_per_step', current_lr, step)
            step = step + 1

        # Update scheduler / learning rate
        scheduler.step()

        final_epoch = epoch + 1 == epochs

        # check the mAP on training set  begin ------------------------------------------------
        if epoch >= cfg.Evaluate_train_start and epoch % cfg.val_interval == 0:
            if args.eval_method == 'coco':
                test_path = 'annotations/instances_train.json'  # .json or .txt btw, should be relative
            elif args.eval_method == 'voc':
                test_path = 'train-ground-truth'
            train_results = evaluate(
                target_size=[cfg.image_size],
                test_path=test_path,
                eval_method=args.eval_method,
                model=model,
                conf=cfg.score_thr,
                device=args.device,
                mode='train')

            if args.eval_method == 'coco':
                train_fitness = train_results[-2]  # Update best mAP
            elif args.eval_method == 'voc':
                train_fitness = train_results[0]  # Update best mAP
            writer.add_scalar('train_mAP', train_fitness, epoch)
        # --------------------------end

        if epoch >= cfg.Evaluate_val_start and epoch % cfg.val_interval == 0:
            if args.eval_method == 'coco':
                test_path = 'annotations/instances_val.json'  # .json or .txt btw, should be relative
            elif args.eval_method == 'voc':
                test_path = 'ground-truth'
            val_results = evaluate(
                target_size=[cfg.image_size],
                test_path=test_path,
                eval_method=args.eval_method,
                model=model,
                conf=cfg.score_thr,
                device=args.device,
                mode='val')

            if args.eval_method == 'coco':
                fitness = val_results[-2]  # Update best mAP
            elif args.eval_method == 'voc':
                fitness = val_results[0]  # Update best mAP

            if fitness > best_fitness:
                best_fitness = fitness

            # write mAP info into tensorboard
            writer.add_scalar('val_mAP', fitness, epoch)

        # save model
        # create checkpoint
        chkpt = {'epoch': epoch,
                 'step': step,
                 'best_fitness': best_fitness,
                 'model': model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel
                 else model.state_dict(),
                 'optimizer': None if final_epoch else optimizer.state_dict(),
                 'scheduler': {'step_count': scheduler._step_count,
                               'last_epoch': scheduler.last_epoch}
                 }

        # save last checkpoint
        # torch.save(chkpt, os.path.join(last_checkpoint_path, 'last.pth'))
        # torch.save(model.state_dict(), os.path.join(checkpoint_path, f'{epoch}.pth')

        # save best checkpoint
        if best_fitness == fitness:
            torch.save(chkpt, os.path.join(best_checkpoint_path, 'best.pth'))

        # save interval checkpoint
        if epoch % cfg.save_interval == 0 and epoch > 0:
            torch.save(chkpt, os.path.join(checkpoint_path, f'{epoch}_{step}.pth'))

    # TensorboardX writer close
    writer.close()


if __name__ == '__main__':
    from utils.utils import show_args, show_cfg
    args = get_args()

    assert args.eval_method in metrics, f'{args.eval_method} unsupported.'

    # show config and args
    show_args(args)
    show_cfg(cfg)

    train(args)
