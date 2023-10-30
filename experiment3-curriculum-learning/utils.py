import os
import math
import argparse
import numpy as np

import torch


def str2bool(v):
    """
    Function to transform strings into booleans.

    v: string variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_best_models(net, optimizer, output_path, best_records, epoch, metric, num_saves=3, track_mean=None):
    if math.isnan(metric):
        metric = 0.0
    if len(best_records) < num_saves:
        best_records.append({'epoch': epoch, 'kappa': metric, 'track_mean': track_mean})

        torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
        # torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))
    else:
        # find min saved acc
        min_index = 0
        for i, r in enumerate(best_records):
            if best_records[min_index]['kappa'] > best_records[i]['kappa']:
                min_index = i
        # print('before', best_records, best_records[min_index], metric,
        #       best_records[min_index]['kappa'], metric > best_records[min_index]['kappa'])
        # check if currect acc is greater than min saved acc
        if metric > best_records[min_index]['kappa']:
            # if it is, delete previous files
            min_step = str(best_records[min_index]['epoch'])

            os.remove(os.path.join(output_path, 'model_' + min_step + '.pth'))
            # os.remove(os.path.join(output_path, 'opt_' + min_step + '.pth'))

            # replace min value with current
            best_records[min_index] = {'epoch': epoch, 'kappa': metric, 'track_mean': track_mean}

            # save current model
            torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
            # torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))
    # print('after', best_records)
    np.save(os.path.join(output_path, 'best_records.npy'), best_records)


def exponent_decay_lr_generator(decay_rate, minimum_lr, batch_to_decay):
    cur_lr = None

    def exponent_decay_lr(initial_lr, batch, history):
        nonlocal cur_lr
        if batch == 0:
            cur_lr = initial_lr
        if (batch % batch_to_decay) == 0 and batch != 0:
            new_lr = cur_lr / decay_rate
            cur_lr = max(new_lr, minimum_lr)
        return cur_lr

    return exponent_decay_lr


def generate_random_batch(indexes, batch_size):
    size_data = indexes.shape[0]
    cur_batch_idxs = np.random.choice(size_data, batch_size, replace=False)
    return indexes[cur_batch_idxs]


class ExponentDecayLRScheduler:
    def __init__(self, optimizer, decay_rate, minimum_lr, batch_to_decay, initial_lr):
        self._optimizer = optimizer
        self.decay_rate = decay_rate
        self.minimum_lr = minimum_lr
        self.batch_to_decay = batch_to_decay
        self.cur_lr = initial_lr

    def step(self, initial_lr, batch):
        if batch == 0:
            self.cur_lr = initial_lr
        if (batch % self.batch_to_decay) == 0 and batch != 0:
            new_lr = self.cur_lr / self.decay_rate
            self.cur_lr = max(new_lr, self.minimum_lr)
        # print('cur_lr', self.cur_lr)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.cur_lr
