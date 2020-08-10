# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch
from tqdm import tqdm
import os

from core.evaluate import accuracy


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, inputs in enumerate(train_loader):
        # measure data loading time

        if torch.cuda.is_available():
            im = inputs['im'].cuda()
            target = inputs['label'].cuda().long().view(-1)

        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # compute output
        output = model(im)
        #target = data['label'].cuda(non_blocking=True).long()

        loss = criterion(output, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), im.size(0))

        prec1 = accuracy(output, target, (1,))

        top1.update(prec1[0].item(), im.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=im.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        if i % 1000 == 0:
            torch.save({
                'epoch': epoch + 1 - 1,
                'model': config.MODEL.NAME,
                'state_dict': model.module.state_dict(),
                'perf': -1000000+i,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(output_dir, 'checkpoint.pth.tar'))


def validate(config, val_loader, model, criterion, output_dir, tb_log_dir,
             writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, inputs in enumerate(tqdm(val_loader)):
            # compute output

            if torch.cuda.is_available():
                im = inputs['im'].cuda()
                target = inputs['label'].cuda().long().view(-1)

            output = model(im)

            #target = target.cuda(non_blocking=True)

            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), im.size(0))
            prec1 = accuracy(output, target, (1,))
            top1.update(prec1[0].item(), im.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Error@1 {error1:.3f}\t' \
              'Accuracy@1 {top1.avg:.3f}\t'.format(
                  batch_time=batch_time, loss=losses, top1=top1,
                  error1=100-top1.avg,)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_top1', top1.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
