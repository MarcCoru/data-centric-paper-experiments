import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import pickle
import datetime

from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from common.model import ResNet18
from common.dfc2020_datamodule import DFC2020DataModule
from common.dfc2020 import DFCDatasetBatchLoader
from predict_dfc_val_test import test
from utils import str2bool, generate_random_batch, ExponentDecayLRScheduler, save_best_models
from common.transforms import get_classification_transform


# logger = TensorBoardLogger("logs/", name="data-centric-exp3")
#
# callbacks = [
#     EarlyStopping(monitor="val_loss", mode="min", patience=10),
#     ModelCheckpoint(
#         dirpath='weights',
#         monitor='val_loss',
#         filename='RN18-epoch{epoch:02d}-val_loss{val_loss:.2f}',
#         save_last=True
#     )
# ]
#
# trainer = pl.Trainer(
#     max_epochs=100,
#     gpus=1,
#     log_every_n_steps=5,
#     fast_dev_run=False,
#     callbacks=callbacks,
#     logger=logger)
#
# trainer.fit(model=model, datamodule=datamodule)


def get_order(dataset_path, weights):
    # creating training data loader
    datamodule = DFC2020DataModule(root=dataset_path, batch_size=256, workers=8)
    datamodule.setup('train')
    train_dataloader = datamodule.train_dataloader()

    # creating model and loading weights
    model = ResNet18(in_channels=10, num_classes=8)
    model.load_state_dict(torch.load(weights)['state_dict'])
    model.cuda()
    model.eval()

    first = True
    all_data = []

    with torch.no_grad():
        # Iterating over batches.
        for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # Obtaining input data, label, path
            image, label, path = data[0], data[1], np.asarray(data[2])
            # print(image.shape, label.shape, type(path), len(path), path[0:2])

            # Casting to cuda variables.
            inps_c = Variable(image).float().cuda()

            # Forwarding.
            logits = model(inps_c)
            # Computing probabilities.
            soft_outs = F.softmax(logits, dim=1).cpu()

            soft_samples = torch.gather(soft_outs, 1, label)

            if first is True:
                all_data = list(zip(path, soft_samples.numpy(), label.cpu().numpy()))
                # print(all_data[0], all_data[1], all_data[-1])
                first = False
            else:
                all_data = np.concatenate((all_data, list(zip(path, soft_samples.numpy(), label.cpu().numpy()))))
                # print(all_data[0], all_data[1], all_data[255])
                # print(all_data[256], all_data[257])
                # print(all_data.shape)

    return all_data


def rank_data_according_to_score(train_scores, reverse=False, random=False):
    res = np.asarray(sorted(range(len(train_scores)), key=lambda k: train_scores[k, 1], reverse=True))

    if reverse:
        res = np.flip(res, 0)
    if random:
        np.random.shuffle(res)

    # print(res.shape)
    # print(res[0:3], train_scores[0:3], train_scores[res[0]], train_scores[res[1]], train_scores[res[2]])
    return res


def balance_order(order, train_scores):
    num_classes = 8

    # size_each_class = len(train_scores) // num_classes  # this gets the average for each class
    bc = np.bincount(train_scores[:, 2].astype(int))
    # this balances using the minimum
    # https://isprs-annals.copernicus.org/articles/V-2-2021/101/2021/isprs-annals-V-2-2021-101-2021.pdf
    size_each_class = np.min(bc[np.nonzero(bc)])
    print('bincount, size per class', bc, size_each_class)

    class_orders = []
    for cls in range(num_classes):
        class_orders.append([i for i in range(len(order)) if train_scores[order[i]][2] == cls])

    new_order = []
    # take each group containing the next easiest image for each class,
    # and putting them according to difficult-level in the new order
    for group_idx in range(size_each_class):
        group = sorted([class_orders[cls][group_idx] for cls in range(num_classes) if class_orders[cls]])
        for idx in group:
            new_order.append(order[idx])

    return new_order


def balance_level_order(order, train_scores, level=100):
    # Get min(total samples, 100) samples for each class
    num_classes = 8

    # size_each_class = len(train_scores) // num_classes  # this gets the average for each class
    bc = np.bincount(train_scores[:, 2].astype(int))
    # this balances using the minimum
    # https://isprs-annals.copernicus.org/articles/V-2-2021/101/2021/isprs-annals-V-2-2021-101-2021.pdf
    smallest_classes = np.nonzero((bc < level).astype(int))
    other_classes = np.nonzero((bc >= level).astype(int))
    print('bincount, index of class < 100 samples, and >= 100', bc, smallest_classes, other_classes)

    # separate the samples per class
    class_orders = []
    for cls in range(num_classes):
        class_orders.append([i for i in range(len(order)) if train_scores[order[i]][2] == cls])

    # for the smallest classes, get all the samples
    new_order = []
    if smallest_classes:
        for scl in smallest_classes[0]:
            for i in class_orders[scl]:
                new_order.append(order[i])

    # take each group containing the next easiest image for each class,
    # and putting them according to difficult-level in the new order
    for group_idx in range(np.max(bc)):
        group = sorted([class_orders[cls][group_idx] for cls in range(num_classes)
                        if cls in other_classes[0] and class_orders[cls] and len(class_orders[cls]) > group_idx])
        for idx in group:
            new_order.append(order[idx])

    print(len(new_order))
    return new_order


def exponent_data_function_generator(order, batches_to_increase, increase_amount, starting_percent):
    cur_percent = 1
    cur_indexes = order
    dataset_size = len(order)

    def data_function(batch):
        nonlocal cur_percent, cur_indexes

        if batch % batches_to_increase == 0:
            if batch == 0:
                percent = starting_percent
            else:
                percent = min(cur_percent * increase_amount, 1)

            if percent != cur_percent:
                cur_percent = percent
                data_limit = int(np.ceil(dataset_size * percent))
                cur_indexes = order[:data_limit]

                # cur_index = train_scores[new_data, :, :, :]
        return np.asarray(cur_indexes)

    return data_function


def val_step(val_loader, net, iter, total_iter):
    # Setting network for evaluation mode.
    net.eval()

    all_labels = None
    all_preds = None
    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(val_loader):

            # Obtaining images, labels and paths for batch.
            inps, labs, _ = data

            # Casting to cuda variables.
            inps_c = Variable(inps).float().cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs = net(inps_c)
            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)

            # Obtaining prior predictions.
            prds = soft_outs.cpu().data.numpy().argmax(axis=1)

            if all_labels is None:
                all_labels = labs
                all_preds = prds
            else:
                all_labels = np.concatenate((all_labels, labs))
                all_preds = np.concatenate((all_preds, prds))

        acc = accuracy_score(all_labels, all_preds)
        conf_m = confusion_matrix(all_labels, all_preds)

        _sum = 0.0
        for k in range(len(conf_m)):
            _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

        print("---- Validation/Test -- Iteration " + str(iter + 1) + "/" + str(total_iter) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
              " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
              )

        sys.stdout.flush()

    return acc, _sum / float(outs.shape[1]), conf_m


def train(dataset_path, train_scores, ranked_index_train_scores,
          learning_rate, lr_decay_rate, minimal_lr, lr_batch_size, weight_decay,
          batch_increase, increase_amount, starting_percent,
          total_iter, batch_size, output_path):
    # training loader
    dataloader = DFCDatasetBatchLoader(dataset_path, train_scores)

    data_index_function = exponent_data_function_generator(ranked_index_train_scores,
                                                           batch_increase, increase_amount, starting_percent)

    # get validation set
    datamodule = DFC2020DataModule(root=dataset_path, batch_size=256, workers=8)
    datamodule.setup('train')
    val_dataloader = datamodule.val_dataloader()

    # creating model
    model = ResNet18(in_channels=10, num_classes=8)
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99))
    # create custom scheduler
    scheduler = ExponentDecayLRScheduler(optimizer, lr_decay_rate, minimal_lr, lr_batch_size, learning_rate)

    best_records = []

    # Average Meter for batch loss.
    train_loss = list()

    # Iterating.
    cur_indexes_size = 500
    cur_lr = learning_rate
    for batch in range(total_iter):
        model.train()

        # get current set of training data
        cur_indexes = data_index_function(batch)
        # print(np.bincount(train_scores[cur_indexes, 2].astype(int)))
        # print('cur_indexes', cur_indexes.shape)
        batch_indexes = generate_random_batch(cur_indexes, batch_size)
        # print('batch_indexes', batch_indexes.shape)
        data, labels = dataloader.get_batch(batch_indexes)

        if cur_indexes_size != len(cur_indexes):
            print('DATA cur, new', cur_indexes_size, len(cur_indexes))
            cur_indexes_size = len(cur_indexes)

        # Casting to cuda variables.
        inps = Variable(data).cuda()
        labs = Variable(labels).cuda()

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs = model(inps)
        soft_outs = F.softmax(outs, dim=1)

        # Obtaining predictions.
        prds = soft_outs.cpu().data.numpy().argmax(axis=1)

        # Computing loss.
        loss = criterion(outs, labs)

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if (batch + 1) % 50 == 0:
            acc = accuracy_score(labels, prds)
            conf_m = confusion_matrix(labels, prds)

            _sum = 0.0
            for k in range(len(conf_m)):
                _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

            print("Training -- Iter " + str(batch + 1) + "/" + str(total_iter) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc) +
                  " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
                  " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
                  )

        if (batch + 1) % 100 == 0:
            # Computing test.
            acc, nacc, cm = val_step(val_dataloader, model, (batch + 1), total_iter)

            save_best_models(model, optimizer, output_path, best_records, epoch=batch, metric=acc, num_saves=3)
        sys.stdout.flush()

        scheduler.step(learning_rate, batch)
        if cur_lr != scheduler.cur_lr:
            print('LR cur, new', cur_lr, scheduler.cur_lr)
            cur_lr = scheduler.cur_lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset_path", type=str, required=False,
                        default='/home/kno/datasets/df2020/', help="Path to the DFC2020 dataset")
    parser.add_argument("--output_path", type=str, required=False,
                        default='output/', help="Path to save outputs")
    parser.add_argument("--weights", type=str, required=False,
                        default="weights/RN18-epochepoch=123-val_lossval_loss=0.23.ckpt",
                        help="Path to the pre-trained weights")
    parser.add_argument("--batch_size", default=256, type=int, help="determine batch size")
    parser.add_argument("--num_epochs", default=600, type=int, help="number of epochs to train on")

    # optimization / learning rate parameters
    parser.add_argument("--learning_rate", "-lr", default=0.001, type=float)
    parser.add_argument("--lr_decay_rate", default=1.5, type=float)
    parser.add_argument("--minimal_lr", default=0.0001, type=float)
    parser.add_argument("--lr_batch_size", default=3000, type=int)
    parser.add_argument("--weight_decay", default=0.00001, type=float)

    # curriculum params
    parser.add_argument("--batch_increase", default=500, type=int,
                        help="interval of batches to increase the amount of data we sample from")
    parser.add_argument("--increase_amount", default=1.5, type=float,
                        help="factor by which we increase the amount of data we sample from")
    parser.add_argument("--starting_percent", default=1000/4000, type=float,  # 1500/4000
                        help="percent of data to sample from in the inital batch")
    parser.add_argument("--balance_classes", type=str, required=False, default='none',
                        choices=['full', 'percentage', 'none'], help='Balance dataset?')

    args = parser.parse_args()
    print(args)

    ts_path = os.path.join(args.dataset_path, 'train_scores.pkl')
    rs_path = os.path.join(args.dataset_path, 'ranked_train_scores.pkl')

    if not os.path.exists(ts_path):
        print('creating scores and ranking')
        train_scores = get_order(args.dataset_path, args.weights)
        ranked_index_train_scores = rank_data_according_to_score(train_scores)

        with open(ts_path, 'wb') as file_pi:
            pickle.dump(train_scores, file_pi)
        with open(rs_path, 'wb') as file_pi:
            pickle.dump(ranked_index_train_scores, file_pi)
    else:
        print('loading scores and ranking')
        with open(ts_path, 'rb') as file_pi:
            train_scores = pickle.load(file_pi)
        with open(rs_path, 'rb') as file_pi:
            ranked_index_train_scores = pickle.load(file_pi)
    print(train_scores.shape, len(ranked_index_train_scores),
          np.min(ranked_index_train_scores), np.max(ranked_index_train_scores),
          train_scores[ranked_index_train_scores[0]], train_scores[ranked_index_train_scores[1]],
          train_scores[ranked_index_train_scores[-1]])

    if args.balance_classes == 'full':
        brs_path = os.path.join(args.dataset_path, 'balanced_ranked_train_scores.pkl')
        if not os.path.exists(brs_path):
            ranked_index_train_scores = balance_order(ranked_index_train_scores, train_scores)
            with open(brs_path, 'wb') as file_pi:
                pickle.dump(ranked_index_train_scores, file_pi)
        else:
            with open(brs_path, 'rb') as file_pi:
                ranked_index_train_scores = pickle.load(file_pi)
        print('balanced full', train_scores.shape, len(ranked_index_train_scores))
    elif args.balance_classes == 'percentage':
        brs_path = os.path.join(args.dataset_path, 'balanced_percentage_ranked_train_scores.pkl')
        if not os.path.exists(brs_path):
            ranked_index_train_scores = balance_level_order(ranked_index_train_scores, train_scores)
            with open(brs_path, 'wb') as file_pi:
                pickle.dump(ranked_index_train_scores, file_pi)
        else:
            with open(brs_path, 'rb') as file_pi:
                ranked_index_train_scores = pickle.load(file_pi)
        print('balanced percentage', train_scores.shape, len(ranked_index_train_scores))

    total_iter = (args.num_epochs * len(ranked_index_train_scores)) // args.batch_size
    print('dataset size', len(ranked_index_train_scores))
    print('total_iter', total_iter)

    train(args.dataset_path, train_scores, ranked_index_train_scores,
          args.learning_rate, args.lr_decay_rate, args.minimal_lr, args.lr_batch_size, args.weight_decay,
          args.batch_increase, args.increase_amount, args.starting_percent,
          total_iter, args.batch_size, args.output_path)
    # test(args.dfc_root_folder, args.val_or_test, args.weights)


