import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorboardX
from tensorboardX import SummaryWriter

import argparse
import os
import pickle
import random
import shutil
import time

import cross_val
import encoders
import gen.feat as featgen
import gen.data as datagen
from graph_sampler import GraphSampler
import load_data
import util

from load_data import read_graphfile
from centrality_nx import cent_calc

torch.manual_seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.cuda.manual_seed_all(7)  # if you are using multi-GPU.
np.random.seed(7)  # Numpy module.
random.seed(7)  # Python random module.
torch.manual_seed(7)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.system('nvidia-smi')


def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " accuracy:", result['acc'])
    return result, labels, preds


def evaluate_attack(dataset, update_adjs, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        # adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        adj = update_adjs[batch_idx]
        h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " accuracy:", result['acc'])
    return result, labels, preds


def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method
    if args.method == 'soft-assign':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio * 100))
        if args.linkpred:
            name += '_lp'
    else:
        name += '_l' + str(args.num_gc_layers)
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    name += '_' + str(args.num_epochs)
    if args.normalize:
        name += '_nor'
    name += '_dp' + str(int(args.dropout * 100))
    if args.pool_concat:
        name += '_pcT'
    else:
        name += '_pcF'
    if args.linkpred:
        name += '_lkT'
    else:
        name += '_lkF'
    return name


def gen_train_plt_name(args):
    return 'results/' + gen_prefix(args) + '.png'


def log_assignment(assign_tensor, writer, epoch, batch_idx):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 6), dpi=300)

    # has to be smaller than args.batch_size
    for i in range(len(batch_idx)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(assign_tensor.cpu().data.numpy()[batch_idx[i]], cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")
    plt.tight_layout()
    fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('assignment', data, epoch)


def log_graph(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 6), dpi=300)

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i + 1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
                edge_color='grey', width=0.5, node_size=300,
                alpha=0.7)
        ax.xaxis.set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs', data, epoch)

    # log a label-less version
    # fig = plt.figure(figsize=(8,6), dpi=300)
    # for i in range(len(batch_idx)):
    #    ax = plt.subplot(2, 2, i+1)
    #    num_nodes = batch_num_nodes[batch_idx[i]]
    #    adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
    #    G = nx.from_numpy_matrix(adj_matrix)
    #    nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color='#336699',
    #            edge_color='grey', width=0.5, node_size=25,
    #            alpha=0.8)

    # plt.tight_layout()
    # fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # writer.add_image('graphs_no_label', data, epoch)

    # colored according to assignment
    assignment = assign_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(8, 6), dpi=300)

    num_clusters = assignment.shape[2]
    all_colors = np.array(range(num_clusters))

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i + 1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()

        label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
        label = label[: batch_num_nodes[batch_idx[i]]]
        node_colors = all_colors[label]

        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=node_colors,
                edge_color='grey', width=0.4, node_size=50, cmap=plt.get_cmap('Set1'),
                vmin=0, vmax=num_clusters - 1,
                alpha=0.8)

    plt.tight_layout()
    fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs_colored', data, epoch)


def save_graph(args, adj, batch_num_nodes, batch_idx, epoch, name='Val', assign_tensor=None):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 6), dpi=300)

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i + 1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
                edge_color='grey', width=0.5, node_size=300,
                alpha=0.7)
        ax.xaxis.set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()

    plt.savefig('results/' + gen_prefix(args) + '_1_' + str(epoch) + '_' + name + '.png', dpi=600)
    plt.close()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # data = tensorboardX.utils.figure_to_image(fig)
    # writer.add_image('graphs', data, epoch)

    # log a label-less version
    # fig = plt.figure(figsize=(8,6), dpi=300)
    # for i in range(len(batch_idx)):
    #    ax = plt.subplot(2, 2, i+1)
    #    num_nodes = batch_num_nodes[batch_idx[i]]
    #    adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
    #    G = nx.from_numpy_matrix(adj_matrix)
    #    nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color='#336699',
    #            edge_color='grey', width=0.5, node_size=25,
    #            alpha=0.8)

    # plt.tight_layout()
    # fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # writer.add_image('graphs_no_label', data, epoch)

    # colored according to assignment
    assignment = assign_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(8, 6), dpi=300)

    num_clusters = assignment.shape[2]
    all_colors = np.array(range(num_clusters))

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i + 1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()

        label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
        label = label[: batch_num_nodes[batch_idx[i]]]
        node_colors = all_colors[label]

        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=node_colors,
                edge_color='grey', width=0.4, node_size=50, cmap=plt.get_cmap('Set1'),
                vmin=0, vmax=num_clusters - 1,
                alpha=0.8)

    plt.tight_layout()
    fig.canvas.draw()

    plt.savefig('results/' + gen_prefix(args) + '_2_' + str(epoch) + '_' + name + '.png', dpi=600)
    plt.close()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # data = tensorboardX.utils.figure_to_image(fig)
    # writer.add_image('graphs_colored', data, epoch)


def data_augmentation(dataset, model, args, val_dataset=None, mask_nodes=True):
    writer_batch_idx = [0, 1]

    train_accs = []
    train_epochs = []
    test_accs = []
    test_epochs = []
    train_update_labels_all = []
    test_update_labels_all = []
    label_attacked_train_all = []
    label_attacked_test_all = []
    train_ratio = []
    test_ratio = []
    asr0_train = []
    asr0_test = []
    asr1_train = []
    asr1_test = []

    model.eval()

    # with open('results/iter/bn_False/train_adj_update.pkl', 'rb') as f:
    #     train_update_adjs_all = pickle.load(f)

    attack_ratio = [0.025, 0.05, 0.075, 0.1]  # , 0.25, 0.3, 0.35, 0.4]
    for ri in range(len(attack_ratio)):
        print('ratio: ', attack_ratio[ri])
        # with open('results/test_adj_update_{}.pkl'.format(ri + 1), 'rb') as f:
        with open('results/test_adj_update_{}.pkl'.format(ri), 'rb') as f:
            test_update_adjs_all = pickle.load(f)

        test_aug_adjs_all = []
        for batch_idx, data in enumerate(val_dataset):
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            test_aug_adjs_batch = test_update_adjs_all[batch_idx].clone()
            for i in range(args.batch_size):
                adj = np.array(test_aug_adjs_batch[i].cpu())
                ct_c, num_edges = cent_calc(adj)
                c_dict = {}
                ct_c = ct_c[0:batch_num_nodes[i], 0:batch_num_nodes[i]]
                for r in range(batch_num_nodes[i]):
                    for c in range(batch_num_nodes[i]):
                        if r < c and ct_c[r][c] != 0:
                            c_dict[(r, c)] = ct_c[r, c].item()
                c_sorted = sorted(c_dict.items(), key=lambda x: x[1], reverse=False)
                num_aug = int(num_edges * attack_ratio[ri])
                print(num_aug)
                for n in range(num_aug):
                    r = c_sorted[n][0][0]
                    c = c_sorted[n][0][1]
                    v = c_sorted[n][1]
                    test_aug_adjs_batch[i][r, c] = 0
                    test_aug_adjs_batch[i][c, r] = 0
            test_aug_adjs_all.append(test_aug_adjs_batch)

        result, true_labels, pred = evaluate_attack(val_dataset, test_aug_adjs_all, model, args, name='Val')
        label_attacked_test_all.append(pred)
        test_accs.append(result['acc'])
        test_ratio.append(attack_ratio[ri])
        asr0_test.append((np.where(label_attacked_test_all[0] != label_attacked_test_all[-1]))[0].shape[0] / 110)
        print(asr0_test[-1])
        if ri == 0:
            asr1_test.append(asr0_test[-1])
        else:
            asr1_test.append((np.where(label_attacked_test_all[-2] != label_attacked_test_all[-1]))[0].shape[0] / 110)
        print(asr1_test[-1])

    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    # plt.plot(train_ratio, train_accs, 'b-', lw=1)  # util.exp_moving_avg(train_accs, 0.85)
    plt.plot(test_ratio, test_accs, 'g:', lw=1)
    # plt.legend(['train', 'test'])
    # print(train_accs)
    print(test_accs)
    # if test_dataset is not None:
    #     plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
    #     plt.legend(['train', 'val', 'test'])
    # else:
    #     plt.plot(best_val_epochs, best_val_accs, 'bo')
    #     plt.legend(['train', 'val'])
    plt.savefig('results/iter/' + gen_prefix(args) + '_acc.png', dpi=600)
    plt.close()

    plt.figure()
    # plt.plot(train_ratio, asr0_train, 'b-', lw=1)
    plt.plot(test_ratio, asr0_test, 'g:', lw=1)
    # plt.legend(['train', 'test'])
    plt.savefig('results/iter/' + gen_prefix(args) + '_asr0.png', dpi=600)

    plt.figure()
    # plt.plot(train_ratio, asr1_train, 'b-', lw=1)
    plt.plot(test_ratio, asr1_test, 'g:', lw=1)
    # plt.legend(['train', 'test'])
    plt.savefig('results/iter/' + gen_prefix(args) + '_asr1.png', dpi=600)
    matplotlib.style.use('default')

    return train_accs, test_accs


def transfer_evaluate(dataset, model, args, val_dataset=None, mask_nodes=True):
    writer_batch_idx = [0, 1]

    train_accs = []
    train_epochs = []
    test_accs = []
    test_epochs = []
    train_update_labels_all = []
    test_update_labels_all = []
    label_attacked_train_all = []
    label_attacked_test_all = []
    train_ratio = []
    test_ratio = []
    asr0_train = []
    asr0_test = []
    asr1_train = []
    asr1_test = []

    model.eval()

    # with open('results/iter/bn_False/train_adj_update.pkl', 'rb') as f:
    #     train_update_adjs_all = pickle.load(f)

    attack_ratio = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    for ri in range(len(attack_ratio)):
        print('ratio: ', attack_ratio[ri])
        # for batch_idx, data in enumerate(dataset):
        #     adj = train_update_adjs_all[ri][batch_idx]
        #     h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
        #     batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
        #     assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()
        #     ypred = model.eval().forward(h0, adj, batch_num_nodes, assign_x=assign_input)
        #
        #     if batch_idx == len(dataset) // 2 and args.method == 'soft-assign':
        #         save_graph(args, adj, batch_num_nodes, writer_batch_idx, ri, 'Train', model.assign_tensor)
        #
        # for batch_idx, data in enumerate(val_dataset):
        #     adj = test_update_adjs_all[ri][batch_idx]
        #     h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
        #     batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
        #     assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()
        #     ypred = model.eval().forward(h0, adj, batch_num_nodes, assign_x=assign_input)
        #
        #     if batch_idx == len(val_dataset) // 2 and args.method == 'soft-assign':
        #         save_graph(args, adj, batch_num_nodes, writer_batch_idx, ri, 'Val', model.assign_tensor)

        # result, true_labels, pred = evaluate_attack(dataset, train_update_adjs_all[ri], model, args, name='Train')
        # label_attacked_train_all.append(pred)
        # train_accs.append(result['acc'])
        # train_ratio.append(attack_ratio[ri])
        # print((np.where(label_attacked_train_all[0] != label_attacked_train_all[-1]))[0].shape[0])
        # asr0_train.append((np.where(label_attacked_train_all[0] != label_attacked_train_all[-1]))[0].shape[0] / 170)
        # print(asr0_train[-1])
        # if ri == 0:
        #     asr1_train.append(asr0_train[-1])
        # else:
        #     asr1_train.append(
        #         (np.where(label_attacked_train_all[-2] != label_attacked_train_all[-1]))[0].shape[0] / 170)
        # print(asr1_train[-1])

        with open('results/iter/bn_False/test_adj_update_{}.pkl'.format(ri), 'rb') as f:
            test_update_adjs_all = pickle.load(f)

        result, true_labels, pred = evaluate_attack(val_dataset, test_update_adjs_all, model, args, name='Val')
        label_attacked_test_all.append(pred)
        test_accs.append(result['acc'])
        test_ratio.append(attack_ratio[ri])
        asr0_test.append((np.where(label_attacked_test_all[0] != label_attacked_test_all[-1]))[0].shape[0] / 110)
        print(asr0_test[-1])
        if ri == 0:
            asr1_test.append(asr0_test[-1])
        else:
            asr1_test.append((np.where(label_attacked_test_all[-2] != label_attacked_test_all[-1]))[0].shape[0] / 110)
        print(asr1_test[-1])

    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    # plt.plot(train_ratio, train_accs, 'b-', lw=1)  # util.exp_moving_avg(train_accs, 0.85)
    plt.plot(test_ratio, test_accs, 'g:', lw=1)
    # plt.legend(['train', 'test'])
    # print(train_accs)
    print(test_accs)
    # if test_dataset is not None:
    #     plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
    #     plt.legend(['train', 'val', 'test'])
    # else:
    #     plt.plot(best_val_epochs, best_val_accs, 'bo')
    #     plt.legend(['train', 'val'])
    plt.savefig('results/iter/' + gen_prefix(args) + '_acc.png', dpi=600)
    plt.close()

    plt.figure()
    # plt.plot(train_ratio, asr0_train, 'b-', lw=1)
    plt.plot(test_ratio, asr0_test, 'g:', lw=1)
    # plt.legend(['train', 'test'])
    plt.savefig('results/iter/' + gen_prefix(args) + '_asr0.png', dpi=600)

    plt.figure()
    # plt.plot(train_ratio, asr1_train, 'b-', lw=1)
    plt.plot(test_ratio, asr1_test, 'g:', lw=1)
    # plt.legend(['train', 'test'])
    plt.savefig('results/iter/' + gen_prefix(args) + '_asr1.png', dpi=600)
    matplotlib.style.use('default')

    return train_accs, test_accs


def train_noiter(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
                 mask_nodes=True):
    writer_batch_idx = [0, 1]

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    # checkpoint = torch.load(dir)
    # model.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(['optimizer'])
    # start_epoch = checkpoint['epoch'] + 1
    # checkpoint = torch.load('models/MUTAG_soft-assign_l3x1_ar10_h64_o64_350_300.pth')
    # optimizer.load_state_dict(checkpoint['optimizer'])

    iter = 0
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    test_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    train_update_adjs_all = []
    test_update_adjs_all = []
    train_update_labels_all = []
    test_update_labels_all = []
    train_true_labels = None
    test_true_labels = None
    train_grads_all = []
    test_grads_all = []

    model.eval()
    ##求梯度
    for batch_idx, data in enumerate(dataset):
        model.zero_grad()
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        # adj = train_update_adjs_all[-1][batch_idx]
        h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
        label = Variable(data['label'].long()).cuda()
        batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        if not args.method == 'soft-assign' or not args.linkpred:
            loss = model.loss(ypred, label)
        else:
            loss = model.loss(ypred, label, adj, batch_num_nodes)

        # for name, parameters in model.named_parameters():
        #     if name == 'adj':
        #         print(parameters.grad)

        loss.backward(retain_graph=True)
        grad = model.update_adj().grad
        if batch_idx == 0:
            print(ypred[0])
            print(grad[0])
        train_grads_batch = []
        for i in range(args.batch_size):
            grad_dict = {}
            g = grad[i][0:batch_num_nodes[i], 0:batch_num_nodes[i]]
            grad_t = g + torch.transpose(g, 0, 1)
            for r in range(batch_num_nodes[i]):
                for c in range(batch_num_nodes[i]):
                    if r < c:
                        grad_dict[(r, c)] = grad_t[r, c].item()
            train_grads_batch.append(sorted(grad_dict.items(), key=lambda x: x[1], reverse=True))
        train_grads_all.append(train_grads_batch)

    for batch_idx, data in enumerate(val_dataset):
        model.zero_grad()
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        # adj = test_update_adjs_all[-1][batch_idx]
        h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
        label = Variable(data['label'].long()).cuda()
        batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        if not args.method == 'soft-assign' or not args.linkpred:
            loss = model.loss(ypred, label)
        else:
            loss = model.loss(ypred, label, adj, batch_num_nodes)

        # for name, parameters in model.named_parameters():
        #     if name == 'adj':
        #         print(parameters.grad)

        loss.backward(retain_graph=True)
        grad = model.update_adj().grad
        test_grads_batch = []
        for i in range(args.batch_size):
            grad_dict = {}
            g = grad[i][0:batch_num_nodes[i], 0:batch_num_nodes[i]]
            grad_t = g + torch.transpose(g, 0, 1)
            for r in range(batch_num_nodes[i]):
                for c in range(batch_num_nodes[i]):
                    if r < c:
                        grad_dict[(r, c)] = grad_t[r, c].item()
            test_grads_batch.append(sorted(grad_dict.items(), key=lambda x: x[1], reverse=True))
        test_grads_all.append(test_grads_batch)

    # attack
    for epoch in range(5):
        train_update_adjs_epoch = []
        test_update_adjs_epoch = []
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            if epoch == 0:
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            else:
                adj = train_update_adjs_all[-1][batch_idx]
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            adj_attack = adj.clone()
            for i in range(args.batch_size):
                updated = False
                for m in range(len(train_grads_all[batch_idx][i])):
                    r = train_grads_all[batch_idx][i][m][0][0]
                    c = train_grads_all[batch_idx][i][m][0][1]
                    v = train_grads_all[batch_idx][i][m][1]
                    if adj_attack[i][r, c] == 0 and v > 0:
                        adj_attack[i][r, c] = 1
                        adj_attack[i][c, r] = 1
                        updated = True
                        # print(v)
                    elif adj_attack[i][r, c] == 1 and v < 0:
                        adj_attack[i][r, c] = 0
                        adj_attack[i][c, r] = 0
                        updated = True
                        # print(v)
                    else:
                        updated = False
                    if updated:
                        break
            train_update_adjs_epoch.append(adj_attack)

            if batch_idx == len(dataset) // 2 and args.method == 'soft-assign':
                save_graph(args, adj_attack, batch_num_nodes, writer_batch_idx, epoch + 1, 'Train', model.assign_tensor)
                if epoch == 0:
                    save_graph(args, adj, batch_num_nodes, writer_batch_idx, 0, 'Train', model.assign_tensor)
                dt = adj_attack - adj
                print(torch.nonzero(dt))

        ##Val
        for batch_idx, data in enumerate(val_dataset):
            if epoch == 0:
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            else:
                adj = test_update_adjs_all[-1][batch_idx]
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            adj_attack = adj.clone()
            for i in range(args.batch_size):
                updated = False
                for m in range(len(test_grads_all[batch_idx][i])):
                    r = test_grads_all[batch_idx][i][m][0][0]
                    c = test_grads_all[batch_idx][i][m][0][1]
                    v = test_grads_all[batch_idx][i][m][1]
                    if adj_attack[i][r, c] == 0 and v > 0:
                        adj_attack[i][r, c] = 1
                        adj_attack[i][c, r] = 1
                        updated = True
                    elif adj_attack[i][r, c] == 1 and v < 0:
                        adj_attack[i][r, c] = 0
                        adj_attack[i][c, r] = 0
                        updated = True
                    else:
                        updated = False
                    if updated:
                        break
            test_update_adjs_epoch.append(adj_attack)

            if batch_idx == len(val_dataset) // 2 and args.method == 'soft-assign':
                save_graph(args, adj_attack, batch_num_nodes, writer_batch_idx, epoch + 1, 'Val', model.assign_tensor)
                if epoch == 0:
                    save_graph(args, adj, batch_num_nodes, writer_batch_idx, 0, 'Val', model.assign_tensor)
                dt = adj_attack - adj
                print(torch.nonzero(dt))

        train_update_adjs_all.append(train_update_adjs_epoch)
        test_update_adjs_all.append(test_update_adjs_epoch)
        if epoch == 0:
            result, train_true_labels, pred = evaluate(dataset, model, args, name='Train', max_num_examples=100)
            train_update_labels_all.append(pred)
            train_accs.append(result['acc'])
            train_epochs.append(epoch)
            result, test_true_labels, pred = evaluate(val_dataset, model, args, max_num_examples=100)
            test_update_labels_all.append(pred)
            test_accs.append(result['acc'])
            test_epochs.append(epoch)
        result, _, pred = evaluate_attack(dataset, train_update_adjs_epoch, model, args, name='Train',
                                          max_num_examples=100)
        train_update_labels_all.append(pred)
        train_accs.append(result['acc'])
        train_epochs.append(epoch + 1)
        result, _, pred = evaluate_attack(val_dataset, test_update_adjs_epoch, model, args, max_num_examples=100)
        test_update_labels_all.append(pred)
        test_accs.append(result['acc'])
        test_epochs.append(epoch + 1)

        print(np.where(train_true_labels != train_update_labels_all[-1]))
        print(np.where(test_true_labels != test_update_labels_all[-1]))

    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), 'b-', lw=1)
    plt.plot(test_epochs, util.exp_moving_avg(test_accs, 0.85), 'g:', lw=1)
    plt.legend(['train', 'test'])
    print(util.exp_moving_avg(train_accs, 0.85))
    print(util.exp_moving_avg(test_accs, 0.85))
    # if test_dataset is not None:
    #     plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
    #     plt.legend(['train', 'val', 'test'])
    # else:
    #     plt.plot(best_val_epochs, best_val_accs, 'bo')
    #     plt.legend(['train', 'val'])
    plt.savefig(gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use('default')

    return model, val_accs


def attack_once(model, args, adj, h0, num_nodes, assign_input, label, ifattack):
    model.eval()
    model.zero_grad()
    ypred = model.eval().forward(h0, adj, num_nodes, assign_x=assign_input)
    if not args.method == 'soft-assign' or not args.linkpred:
        loss = model.loss(ypred, label)
    else:
        loss = model.loss(ypred, label, adj, num_nodes)

    loss.backward(retain_graph=True)
    grad = model.update_adj().grad
    adj_attacked = adj.clone()
    for i in range(adj.size()[0]):
        if ifattack[i] == False:
            continue
        grad_dict = {}
        g = grad[i][0:num_nodes[i], 0:num_nodes[i]]
        grad_t = g + torch.transpose(g, 0, 1)
        for r in range(num_nodes[i]):
            for c in range(num_nodes[i]):
                if r < c:
                    grad_dict[(r, c)] = grad_t[r, c].item()
        grad_sorted = sorted(grad_dict.items(), key=lambda x: x[1], reverse=True)

        updated = False
        for m in range(len(grad_sorted)):
            r = grad_sorted[m][0][0]
            c = grad_sorted[m][0][1]
            v = grad_sorted[m][1]
            if adj_attacked[i][r, c] == 0 and v > 0:
                adj_attacked[i][r, c] = 1
                adj_attacked[i][c, r] = 1
                updated = True
                # print(v)
            elif adj_attacked[i][r, c] == 1 and v < 0:
                adj_attacked[i][r, c] = 0
                adj_attacked[i][c, r] = 0
                updated = True
                # print(v)
            else:
                updated = False
            if updated == True:
                break
    return adj_attacked, model.assign_tensor


def attack_iter(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
                mask_nodes=True):
    writer_batch_idx = [0, 1]
    model.eval()

    label_attacked_train_all = []
    train_accs = []
    train_ratio = []
    adj_attacked_train_all = []
    num_edges_train = []
    asr0_train = []
    asr1_train = []

    label_attacked_test_all = []
    test_accs = []
    test_ratio = []
    adj_attacked_test_all = []
    num_edges_test = []
    asr0_test = []
    asr1_test = []

    attack_ratio = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    for ri in range(len(attack_ratio)):
        # print('ratio: ', attack_ratio[ri])
        # # for Train
        # adj_attacked_train_per_ratio = []
        # for batch_idx, data in enumerate(dataset):
        #     if ri == 0:
        #         adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        #         num_edges_train.append(data['num_edges'].int().numpy())
        #     else:
        #         adj = adj_attacked_train_all[-1][batch_idx]
        #     h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
        #     label = Variable(data['label'].long()).cuda()
        #     batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
        #     assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()
        #     if ri == 0:
        #         attack_num = (num_edges_train[batch_idx]*attack_ratio[ri]).astype(np.int16)
        #     else:
        #         attack_num = (num_edges_train[batch_idx]*attack_ratio[ri]).astype(np.int16)\
        #                      -(num_edges_train[batch_idx]*attack_ratio[ri-1]).astype(np.int16)
        #     adj_attacked = adj.clone()
        #     for n in range(np.max(attack_num)):
        #         ifattack = []
        #         for bi in range(args.batch_size):
        #             if n<attack_num[bi]:
        #                 ifattack.append(True)
        #             else:
        #                 ifattack.append(False)
        #         adj_attacked, assign_tensor = attack_once(model, args, adj_attacked, h0,
        #                                                   batch_num_nodes, assign_input, label, ifattack)
        #
        #     if batch_idx == len(dataset) // 2 and args.method == 'soft-assign':
        #         print('num_edges:', num_edges_train[batch_idx])
        #         print('ratio: ', attack_ratio[ri])
        #         print('attack_num:', attack_num)
        #         if ri == 0:
        #             ypred = model.eval().forward(h0, adj, batch_num_nodes, assign_x=assign_input)
        #             save_graph(args, adj, batch_num_nodes, writer_batch_idx, ri, 'Train', model.assign_tensor)
        #         else:
        #             save_graph(args, adj_attacked, batch_num_nodes, writer_batch_idx, ri, 'Train', assign_tensor)
        #         dt = adj_attacked - adj
        #         print(torch.nonzero(dt))
        #
        #     adj_attacked_train_per_ratio.append(adj_attacked)
        # adj_attacked_train_all.append(adj_attacked_train_per_ratio)

        # for Test
        adj_attacked_test_per_ratio = []
        for batch_idx, data in enumerate(val_dataset):
            if ri == 0:
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                num_edges_test.append(data['num_edges'].int().numpy())
            else:
                adj = adj_attacked_test_all[-1][batch_idx]
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()
            if ri == 0:
                attack_num = (
                        [batch_idx] * attack_ratio[ri]).astype(np.int16)
            else:
                attack_num = (num_edges_test[batch_idx] * attack_ratio[ri]).astype(np.int16) \
                             - (num_edges_test[batch_idx] * attack_ratio[ri - 1]).astype(np.int16)
            adj_attacked = adj.clone()
            for n in range(np.max(attack_num)):
                ifattack = []
                for bi in range(adj.size()[0]):
                    if n < attack_num[bi]:
                        ifattack.append(True)
                    else:
                        ifattack.append(False)
                adj_attacked, assign_tensor = attack_once(model, args, adj_attacked, h0,
                                                          batch_num_nodes, assign_input, label, ifattack)

            if batch_idx == len(val_dataset) // 2 and args.method == 'soft-assign':
                # print('num_edges:', num_edges_test[batch_idx])
                # print('ratio: ', attack_ratio[ri])
                # print('attack_num:', attack_num)
                if ri == 0:
                    ypred = model.eval().forward(h0, adj, batch_num_nodes, assign_x=assign_input)
                    save_graph(args, adj, batch_num_nodes, writer_batch_idx, ri, 'Val', model.assign_tensor)
                else:
                    save_graph(args, adj_attacked, batch_num_nodes, writer_batch_idx, ri, 'Val', assign_tensor)
                dt = adj_attacked - adj
                # print(torch.nonzero(dt))

            adj_attacked_test_per_ratio.append(adj_attacked)

        rth = os.path.join(args.logdir, gen_prefix(args), 'test_adj_update_{}.pkl'.format(ri))
        with open(rth, 'wb') as f:
            pickle.dump(adj_attacked_test_per_ratio, f)
        adj_attacked_test_all.append(adj_attacked_test_per_ratio)

        # #for train
        # if ri == 0:
        #     result, true_labels, pred = evaluate(dataset, model, args, name='Train')
        # else:
        #     result, true_labels, pred = evaluate_attack(dataset, adj_attacked_train_per_ratio, model, args, name='Train')
        # label_attacked_train_all.append(pred)
        # train_accs.append(result['acc'])
        # train_ratio.append(attack_ratio[ri])
        # print((np.where(label_attacked_train_all[0] != label_attacked_train_all[-1]))[0].shape[0])
        # asr0_train.append((np.where(label_attacked_train_all[0] != label_attacked_train_all[-1]))[0].shape[0]/170)
        # print(asr0_train[-1])
        # if ri == 0:
        #     asr1_train.append(asr0_train[-1])
        # else:
        #     asr1_train.append((np.where(label_attacked_train_all[-2] != label_attacked_train_all[-1]))[0].shape[0]/170)
        # print(asr1_train[-1])

        # for test
        if ri == 0:
            result, true_labels, pred = evaluate(val_dataset, model, args, name='Val')
        else:
            result, true_labels, pred = evaluate_attack(val_dataset, adj_attacked_test_per_ratio, model, args,
                                                        name='Val')
        label_attacked_test_all.append(pred)
        test_accs.append(result['acc'])
        test_ratio.append(attack_ratio[ri])
        asr0_test.append((np.where(label_attacked_test_all[0] != label_attacked_test_all[-1]))[0].shape[0] / 110)
        print(asr0_test[-1])
        if ri == 0:
            asr1_test.append(asr0_test[-1])
        else:
            asr1_test.append((np.where(label_attacked_test_all[-2] != label_attacked_test_all[-1]))[0].shape[0] / 110)
        print(asr1_test[-1])

    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    # plt.plot(train_ratio, train_accs, 'b-', lw=1)  # util.exp_moving_avg(train_accs, 0.85)
    plt.plot(test_ratio, test_accs, 'g:', lw=1)
    # plt.legend(['train', 'test'])
    # print(train_accs)
    print(test_accs)
    # if test_dataset is not None:
    #     plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
    #     plt.legend(['train', 'val', 'test'])
    # else:
    #     plt.plot(best_val_epochs, best_val_accs, 'bo')
    #     plt.legend(['train', 'val'])
    plt.savefig('results/iter/' + gen_prefix(args) + '_acc.png', dpi=600)
    plt.close()

    # plt.figure()
    # plt.plot(test_ratio, asr0_test, 'g:', lw=1)
    # plt.savefig('results/iter/' + gen_prefix(args) + '_asr0.png', dpi=600)

    # plt.figure()
    # plt.plot(test_ratio, asr1_test, 'g:', lw=1)
    # plt.savefig('results/iter/' + gen_prefix(args) + '_asr1.png', dpi=600)
    # matplotlib.style.use('default')

    # save adj_update
    # with open('results/iter/bn_False/train_adj_update.pkl', 'wb') as f:
    #     pickle.dump(adj_attacked_train_all, f)
    # rth = os.path.join(args.logdir, gen_prefix(args), 'test_adj_update.pth')
    # with open(rth, 'wb') as f:
    # pickle.dump(adj_attacked_test_all, f)
    print('adj is saved!')
    return model, test_accs


def grad_order(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
               mask_nodes=True):
    model.eval()

    writer_batch_idx = [0, 3, 6, 9]

    adj_grads = dict()
    adj_grads['adj'] = []
    adj_grads['grad'] = []

    for epoch in range(1):
        for batch_idx, data in enumerate(val_dataset):
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if not args.method == 'soft-assign' or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward(retain_graph=True)
            grads = model.update_adj().grad
            for i in range(adj.size()[0]):
                g = grads[i][0:batch_num_nodes[i], 0:batch_num_nodes[i]]
                grad_t = g + torch.transpose(g, 0, 1)
                adj_grads['grad'].append(np.array(grad_t.cpu()) / 2.)
                adj_grads['adj'].append(np.array(adj[i].cpu()))
            elapsed = time.time() - begin_time

        agth = os.path.join(args.logdir, gen_prefix(args), 'val_adj_grads.pkl')
        with open(agth, 'wb') as f:
            pickle.dump(adj_grads, f)

    return adj_grads


def attack_random(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
                  mask_nodes=True):
    writer_batch_idx = [0, 1]
    model.eval()
    label_attacked_train_all = []
    train_accs = []
    train_ratio = []
    adj_attacked_train_all = []
    num_edges_train = []
    asr0_train = []
    asr1_train = []

    label_attacked_test_all = []
    test_accs = []
    test_ratio = []
    adj_attacked_test_all = []
    num_edges_test = []
    asr0_test = []
    asr1_test = []

    attack_ratio = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    for ri in range(len(attack_ratio)):
        # for Test
        adj_attacked_test_per_ratio = []
        for batch_idx, data in enumerate(val_dataset):
            if ri == 0:
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                num_edges_test.append(data['num_edges'].int().numpy())
            else:
                adj = adj_attacked_test_all[-1][batch_idx]
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()
            if ri == 0:
                attack_num = (num_edges_test[batch_idx] * attack_ratio[ri]).astype(np.int16)
            else:
                attack_num = (num_edges_test[batch_idx] * attack_ratio[ri]).astype(np.int16) \
                             - (num_edges_test[batch_idx] * attack_ratio[ri - 1]).astype(np.int16)
            adj_attacked = adj.clone()
            for n in range(np.max(attack_num)):
                for bi in range(adj.size()[0]):
                    if n < attack_num[bi]:
                        r = random.randint(0, batch_num_nodes[bi] - 1)
                        c = random.randint(0, batch_num_nodes[bi] - 1)
                        if adj_attacked[bi][r, c] == 0:
                            adj_attacked[bi][r, c] = 1
                            adj_attacked[bi][c, r] = 1
                        else:
                            adj_attacked[bi][r, c] = 0
                            adj_attacked[bi][c, r] = 0

            # if batch_idx == len(val_dataset) // 2 and args.method == 'soft-assign':
            # print('num_edges:', num_edges_test[batch_idx])
            # print('ratio: ', attack_ratio[ri])
            # print('attack_num:', attack_num)
            # if ri == 0:
            # ypred = model.eval().forward(h0, adj, batch_num_nodes, assign_x=assign_input)
            # save_graph(args, adj, batch_num_nodes, writer_batch_idx, ri, 'Val', model.assign_tensor)
            # else:
            # ypred = model.eval().forward(h0, adj, batch_num_nodes, assign_x=assign_input)
            # save_graph(args, adj_attacked, batch_num_nodes, writer_batch_idx, ri, 'Val', model.assign_tensor)
            # dt = adj_attacked - adj
            # print(torch.nonzero(dt))

            adj_attacked_test_per_ratio.append(adj_attacked)
        rth = os.path.join(args.logdir, 'random', 'test_adj_update_{}.pkl'.format(ri))
        with open(rth, 'wb') as f:
            pickle.dump(adj_attacked_test_per_ratio, f)
        adj_attacked_test_all.append(adj_attacked_test_per_ratio)

        # for test
        if ri == 0:
            result, true_labels, pred = evaluate(val_dataset, model, args, name='Val')
        else:
            result, true_labels, pred = evaluate_attack(val_dataset, adj_attacked_test_per_ratio, model, args,
                                                        name='Val')
        label_attacked_test_all.append(pred)
        test_accs.append(result['acc'])
        test_ratio.append(attack_ratio[ri])

    print(test_accs)
    print(test_ratio)
    print('adj is saved!')

    return model, test_accs


def train_iter(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
               mask_nodes=True):
    writer_batch_idx = [0, 1]
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    # checkpoint = torch.load(dir)
    # model.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(['optimizer'])
    # start_epoch = checkpoint['epoch'] + 1
    # checkpoint = torch.load('models/MUTAG_soft-assign_l3x1_ar10_h64_o64_350_300.pth')
    # optimizer.load_state_dict(checkpoint['optimizer'])

    train_accs = []
    train_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    train_update_adjs_all = []
    test_update_adjs_all = []
    train_update_labels_all = []
    test_update_labels_all = []
    train_true_labels = None
    test_true_labels = None
    train_grads_all = []
    test_grads_all = []

    model.eval()
    ave_num_edge = 20

    # ##求梯度
    # attack
    for epoch in range(ave_num_edge):
        train_grads_epoch = []
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            # adj = train_update_adjs_all[-1][batch_idx]
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()
            ypred = model.eval().forward(h0, adj, batch_num_nodes, assign_x=assign_input)

            if not args.method == 'soft-assign' or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)

            loss.backward(retain_graph=True)
            grad = model.update_adj().grad
            train_grads_batch = []
            for i in range(args.batch_size):
                grad_dict = {}
                g = grad[i][0:batch_num_nodes[i], 0:batch_num_nodes[i]]
                grad_t = g + torch.transpose(g, 0, 1)
                for r in range(batch_num_nodes[i]):
                    for c in range(batch_num_nodes[i]):
                        if r < c:
                            grad_dict[(r, c)] = grad_t[r, c].item()
                train_grads_batch.append(sorted(grad_dict.items(), key=lambda x: x[1], reverse=True))
            train_grads_epoch.append(train_grads_batch)
        train_grads_all.append(train_grads_epoch)

        # val
        test_grads_epoch = []
        for batch_idx, data in enumerate(val_dataset):
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            # adj = test_update_adjs_all[-1][batch_idx]
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if not args.method == 'soft-assign' or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)

            # for name, parameters in model.named_parameters():
            #     if name == 'adj':
            #         print(parameters.grad)

            loss.backward(retain_graph=True)
            grad = model.update_adj().grad
            test_grads_batch = []
            for i in range(args.batch_size):
                grad_dict = {}
                g = grad[i][0:batch_num_nodes[i], 0:batch_num_nodes[i]]
                grad_t = g + torch.transpose(g, 0, 1)
                for r in range(batch_num_nodes[i]):
                    for c in range(batch_num_nodes[i]):
                        if r < c:
                            grad_dict[(r, c)] = grad_t[r, c].item()
                test_grads_batch.append(sorted(grad_dict.items(), key=lambda x: x[1], reverse=True))
            test_grads_epoch.append(test_grads_batch)
        test_grads_all.append(test_grads_epoch)
        # print(test_grads_epoch)

        print('Epoch: ', epoch)
        train_update_adjs_epoch = []
        for batch_idx, data in enumerate(dataset):
            if epoch == 0:
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            else:
                adj = train_update_adjs_all[-1][batch_idx]
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            adj_attack = adj.clone()
            for i in range(args.batch_size):
                updated = False
                for m in range(len(train_grads_epoch[batch_idx][i])):
                    r = train_grads_epoch[batch_idx][i][m][0][0]
                    c = train_grads_epoch[batch_idx][i][m][0][1]
                    v = train_grads_epoch[batch_idx][i][m][1]
                    if adj_attack[i][r, c] == 0 and v > 0:
                        adj_attack[i][r, c] = 1
                        adj_attack[i][c, r] = 1
                        updated = True
                        # print(v)
                    elif adj_attack[i][r, c] == 1 and v < 0:
                        adj_attack[i][r, c] = 0
                        adj_attack[i][c, r] = 0
                        updated = True
                        # print(v)
                    else:
                        updated = False
                    if updated:
                        break
            train_update_adjs_epoch.append(adj_attack)

            if batch_idx == len(dataset) // 2 and args.method == 'soft-assign':
                save_graph(args, adj_attack, batch_num_nodes, writer_batch_idx, epoch + 1, 'Train', model.assign_tensor)
                if epoch == 0:
                    save_graph(args, adj, batch_num_nodes, writer_batch_idx, 0, 'Train', model.assign_tensor)
                dt = adj_attack - adj
                print(torch.nonzero(dt))
        train_update_adjs_all.append(train_update_adjs_epoch)

        ##Val
        test_update_adjs_epoch = []
        for batch_idx, data in enumerate(val_dataset):
            if epoch == 0:
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            else:
                adj = test_update_adjs_all[-1][batch_idx]
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            adj_attack = adj.clone()
            for i in range(args.batch_size):
                updated = False
                for m in range(len(test_grads_epoch[batch_idx][i])):
                    r = test_grads_epoch[batch_idx][i][m][0][0]
                    c = test_grads_epoch[batch_idx][i][m][0][1]
                    v = test_grads_epoch[batch_idx][i][m][1]
                    if adj_attack[i][r, c] == 0 and v > 0:
                        adj_attack[i][r, c] = 1
                        adj_attack[i][c, r] = 1
                        updated = True
                    elif adj_attack[i][r, c] == 1 and v < 0:
                        adj_attack[i][r, c] = 0
                        adj_attack[i][c, r] = 0
                        updated = True
                    else:
                        updated = False
                    if updated:
                        break
            test_update_adjs_epoch.append(adj_attack)

            if batch_idx == len(val_dataset) // 2 and args.method == 'soft-assign':
                save_graph(args, adj_attack, batch_num_nodes, writer_batch_idx, epoch + 1, 'Val', model.assign_tensor)
                if epoch == 0:
                    save_graph(args, adj, batch_num_nodes, writer_batch_idx, 0, 'Val', model.assign_tensor)
                dt = adj_attack - adj
                print(torch.nonzero(dt))
        test_update_adjs_all.append(test_update_adjs_epoch)

        if epoch == 0:
            result, train_true_labels, pred = evaluate(dataset, model, args, name='Train')
            train_update_labels_all.append(pred)
            train_accs.append(result['acc'])
            train_epochs.append(epoch)
            result, test_true_labels, pred = evaluate(val_dataset, model, args)
            test_update_labels_all.append(pred)
            test_accs.append(result['acc'])
            test_epochs.append(epoch)
        result, _, pred = evaluate_attack(dataset, train_update_adjs_epoch, model, args, name='Train')
        train_update_labels_all.append(pred)
        train_accs.append(result['acc'])
        train_epochs.append(epoch + 1)
        result, _, pred = evaluate_attack(val_dataset, test_update_adjs_epoch, model, args)
        test_update_labels_all.append(pred)
        test_accs.append(result['acc'])
        test_epochs.append(epoch + 1)

        print(np.where(train_true_labels != train_update_labels_all[-1]))
        print(np.where(test_true_labels != test_update_labels_all[-1]))

    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(train_epochs, train_accs, 'b-', lw=1)  # util.exp_moving_avg(train_accs, 0.85)
    plt.plot(test_epochs, test_accs, 'g:', lw=1)
    plt.legend(['train', 'test'])
    print(train_accs)
    print(test_accs)
    # if test_dataset is not None:
    #     plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
    #     plt.legend(['train', 'val', 'test'])
    # else:
    #     plt.plot(best_val_epochs, best_val_accs, 'bo')
    #     plt.legend(['train', 'val'])
    plt.savefig(gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use('default')

    # save adj_update
    with open('results/iter/train_adj_update.pkl', 'wb') as f:
        pickle.dump(train_update_adjs_all, f)
    with open('results/iter/test_adj_update.pkl', 'wb') as f:
        pickle.dump(test_update_adjs_all, f)
    print('adj is saved!')

    return model, val_accs


def prepare_data(graphs, args, test_graphs=None, max_nodes=0):
    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs),
          '; Num testing graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
          max([G.number_of_nodes() for G in graphs]), ', '
                                                      "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
          ', '
          "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
           dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim


def syn_community1v2(args, writer=None, export_graphs=False):
    # data
    graphs1 = datagen.gen_ba(range(40, 60), range(4, 5), 500,
                             featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)))
    for G in graphs1:
        G.graph['label'] = 0
    if export_graphs:
        util.draw_graph_list(graphs1[:16], 4, 4, 'figs/ba')

    graphs2 = datagen.gen_2community_ba(range(20, 30), range(4, 5), 500, 0.3,
                                        [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))])
    for G in graphs2:
        G.graph['label'] = 1
    if export_graphs:
        util.draw_graph_list(graphs2[:16], 4, 4, 'figs/ba2')

    graphs = graphs1 + graphs2

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(graphs, args)
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                                           args.num_gc_layers, bn=args.bn).cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                                         args.num_gc_layers, bn=args.bn).cuda()

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
          writer=writer)


def syn_community2hier(args, writer=None):
    # data
    feat_gen = [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))]
    graphs1 = datagen.gen_2hier(1000, [2, 4], 10, range(4, 5), 0.1, 0.03, feat_gen)
    graphs2 = datagen.gen_2hier(1000, [3, 3], 10, range(4, 5), 0.1, 0.03, feat_gen)
    graphs3 = datagen.gen_2community_ba(range(28, 33), range(4, 7), 1000, 0.25, feat_gen)

    for G in graphs1:
        G.graph['label'] = 0
    for G in graphs2:
        G.graph['label'] = 1
    for G in graphs3:
        G.graph['label'] = 2

    graphs = graphs1 + graphs2 + graphs3

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(graphs, args)

    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, linkpred=args.linkpred, args=args, assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                                           args.num_gc_layers, bn=args.bn, args=args,
                                           assign_input_dim=assign_input_dim).cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                                         args.num_gc_layers, bn=args.bn, args=args).cuda()
    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
          writer=writer)


def pkl_task(args, feat=None):
    with open(os.path.join(args.datadir, args.pkl_fname), 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    graphs = data[0]
    labels = data[1]
    test_graphs = data[2]
    test_labels = data[3]

    for i in range(len(graphs)):
        graphs[i].graph['label'] = labels[i]
    for i in range(len(test_graphs)):
        test_graphs[i].graph['label'] = test_labels[i]

    if feat is None:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
        for G in test_graphs:
            featgen_const.gen_node_features(G)

    train_dataset, test_dataset, max_num_nodes = prepare_data(graphs, args, test_graphs=test_graphs)
    model = encoders.GcnEncoderGraph(
        args.input_dim, args.hidden_dim, args.output_dim, args.num_classes,
        args.num_gc_layers, bn=args.bn).cuda()
    train(train_dataset, model, args, test_dataset=test_dataset)
    evaluate(test_dataset, model, args, 'Validation')


def benchmark_task(args, writer=None, feat='node-label'):
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in graphs[0].node[0]:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = np.array(G.node[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = \
        prepare_data(graphs, args, max_nodes=args.max_nodes)
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool, concat=True,
            bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
            assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
          writer=writer)
    evaluate(test_dataset, model, args, 'Validation')


def benchmark_task_val(args, writer=None, feat='node-label'):
    all_vals = []
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in graphs[0].node[0]:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = np.array(G.node[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    for i in range(1):  # 10
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
            cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)
        if args.method == 'soft-assign':
            print('Method: soft-assign')
            model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes,
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool, gcn_concat=True,
                pool_concat=args.pool_concat,  # gcn_concat: dense or ..    #pool_concat: our is True
                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                assign_input_dim=assign_input_dim).cuda()
        elif args.method == 'base-set2set':
            print('Method: base-set2set')
            model = encoders.GcnSet2SetEncoder(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()
        else:
            print('Method: base')
            model = encoders.GcnEncoderGraph(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()

        mth = os.path.join(args.logdir, gen_prefix(args), '200.pth')
        checkpoint = torch.load(mth)
        model.load_state_dict(checkpoint['net'])
        # optimizer.load_state_dict(['optimizer'])
        # start_epoch = checkpoint['epoch'] + 1

        # for name, parameters in model.named_parameters():
        #     print(name, ':', parameters.size())
        #     if name == 'adj':
        #         parameters.requires_grad=False

        # for name, parameters in model.named_parameters():
        #     print(name, ':', parameters.size())
        #     if name != 'adj':
        #         parameters.requires_grad=False
        #     else:
        #         parameters.requires_grad = True
        # parm[name] = parameters.detach().numpy()

        _, val_accs = attack_iter(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
                                  writer=writer)
        # adj_grads = grad_order(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
        #     writer=writer)
        # _, val_accs = attack_random(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
        #                     writer=writer)
        # train_accs, val_accs = transfer_evaluate(train_dataset, model, args, val_dataset=val_dataset)
        # train_accs, val_accs = data_augmentation(train_dataset, model, args, val_dataset=val_dataset)

        # all_vals.append(np.array(val_accs))
    # all_vals = np.vstack(all_vals)
    # all_vals = np.mean(all_vals, axis=0)
    # print(all_vals)
    # print(np.max(all_vals))
    # print(np.argmax(all_vals))


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
                                  help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
                           help='Name of the pkl data file')
    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
                                 help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
                                 help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=False,
                        help='Whether link prediction side objective is used')

    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
                        help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')

    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
                        help='suffix added to the output filename')

    parser.set_defaults(datadir='data',
                        logdir='log',
                        dataset='bmname',
                        bmname='PROTEINS',
                        max_nodes=700,
                        cuda='1',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=10,  # 20
                        num_epochs=201,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=64,
                        output_dim=64,
                        num_classes=2,  # yao gai
                        num_gc_layers=3,
                        dropout=0.1,
                        bn=False,
                        method='soft-assign',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1,
                        normalize=False,
                        pool_concat=True
                        )
    return parser.parse_args()


def main():
    prog_args = arg_parse()

    # export scalar data to JSON for external processing
    path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    # if os.path.isdir(path):
    # print('Remove existing log dir: ', path)
    # shutil.rmtree(path)
    writer = SummaryWriter(path)
    # writer = None

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)

    if prog_args.bmname is not None:
        benchmark_task_val(prog_args, writer=writer)
    elif prog_args.pkl_fname is not None:
        pkl_task(prog_args)
    elif prog_args.dataset is not None:
        if prog_args.dataset == 'syn1v2':
            syn_community1v2(prog_args, writer=writer)
        if prog_args.dataset == 'syn2hier':
            syn_community2hier(prog_args, writer=writer)
        if prog_args.dataset == 'bmname':
            benchmark_task(prog_args, writer=writer)

    writer.close()


if __name__ == "__main__":
    # main()
    args = arg_parse()
    path = os.path.join(args.logdir, gen_prefix(args))
    writer = SummaryWriter(path)
    # writer = None
    feat = 'node-label'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    print('CUDA', args.cuda)

    all_vals = []
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in graphs[0].node[0]:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = np.array(G.node[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    for i in range(1):  # 10
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
            cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)

        model = encoders.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool, gcn_concat=True,
            pool_concat=args.pool_concat,  # gcn_concat: dense or ..    #pool_concat: our is True
            bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
            assign_input_dim=assign_input_dim).cuda()

        mth = os.path.join(args.logdir, gen_prefix(args), '200.pth')
        checkpoint = torch.load(mth)
        model.load_state_dict(checkpoint['net'])

        # _, val_accs = attack_iter(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
        #                           writer=writer)
        writer_batch_idx = [0, 1]
        model.eval()

        label_attacked_train_all = []
        # train_accs = []
        train_ratio = []
        adj_attacked_train_all = []
        num_edges_train = []
        asr0_train = []
        asr1_train = []

        label_attacked_test_all = []
        test_accs = []
        test_ratio = []
        adj_attacked_test_all = []
        num_edges_test = []
        ASR = []

        # train_accs, val_accs = data_augmentation(train_dataset, model, args, val_dataset=val_dataset)
        #
        #
        # attack_ratio = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        # for ri in range(len(attack_ratio)):
        #
        #
        #     with open('results/test_adj_update_{}.pkl'.format(ri), 'rb') as f:
        #         test_update_adjs_all = pickle.load(f)
        #         adj_attacked_test_per_ratio = test_update_adjs_all
        #
        #
        #
        #
        #
        #     # for test
        #     if ri == 0:
        #         result, true_labels, pred_ori = evaluate(val_dataset, model, args, name='Val_ori')
        #         # _, _, pred_ori = evaluate(val_dataset, model, args, name='Val_ori')
        #         _, _, pred = evaluate_attack(val_dataset, adj_attacked_test_per_ratio, model, args,
        #                                                     name='Val_att')
        #     else:
        #         result, true_labels, pred_ori = evaluate(val_dataset, model, args, name='Val-ori')
        #         _, _, pred = evaluate_attack(val_dataset, adj_attacked_test_per_ratio, model, args,
        #                                                     name='Val_att')
        #     label_attacked_test_all.append(pred)
        #     test_accs.append(result['acc'])
        #     test_ratio.append(attack_ratio[ri])
        #     # a = np.where(pred != true_labels)
        #
        #     suc_num = 0
        #     for i in range(len(pred)):
        #         if pred[i] != true_labels[i] and pred_ori[i] == true_labels[i]:
        #             suc_num +=1
        #     ASR.append(suc_num /(np.where(pred_ori == true_labels))[0].shape[0])
        #     print(" ASR :",ASR[-1])
        att = 8
        attack_ratio = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # , 0.25, 0.3, 0.35, 0.4]
        for ri in range(len(attack_ratio)):
            print('compression_ratio: ', attack_ratio[ri])
            # with open('results/test_adj_update_{}.pkl'.format(ri + 1), 'rb') as f:
            with open('results/test_adj_update_{}.pkl'.format(8), 'rb') as f:
                test_update_adjs_all = pickle.load(f)

            test_aug_adjs_all = []
            for batch_idx, data in enumerate(val_dataset):
                batch_num_nodes = data['num_nodes'].int().numpy() if True else None
                test_aug_adjs_batch = test_update_adjs_all[batch_idx].clone()
                for i in range(args.batch_size):
                    adj = np.array(test_aug_adjs_batch[i].cpu())
                    ct_c, num_edges = cent_calc(adj)
                    c_dict = {}
                    ct_c = ct_c[0:batch_num_nodes[i], 0:batch_num_nodes[i]]
                    for r in range(batch_num_nodes[i]):
                        for c in range(batch_num_nodes[i]):
                            if r < c and ct_c[r][c] != 0:
                                c_dict[(r, c)] = ct_c[r, c].item()
                    c_sorted = sorted(c_dict.items(), key=lambda x: x[1], reverse=False)
                    num_aug = int(num_edges * attack_ratio[ri])
                    print(att,":  ","compression_num:", num_aug)
                    for n in range(num_aug):
                        r = c_sorted[n][0][0]
                        c = c_sorted[n][0][1]
                        v = c_sorted[n][1]
                        test_aug_adjs_batch[i][r, c] = 0
                        test_aug_adjs_batch[i][c, r] = 0
                test_aug_adjs_all.append(test_aug_adjs_batch)

            _, _, pred_ori = evaluate(val_dataset, model, args, name='Val-ori')
            result, true_labels, pred = evaluate_attack(val_dataset, test_aug_adjs_all, model, args,
                                                        name='Val-compression')
            label_attacked_test_all.append(pred)
            test_accs.append(result['acc'])
            test_ratio.append(attack_ratio[ri])

            suc_num = 0
            for i in range(len(pred)):
                if pred[i] != true_labels[i] and pred_ori[i] == true_labels[i]:
                    suc_num += 1
            ASR.append(suc_num / (np.where(pred_ori == true_labels))[0].shape[0])
            print(" ASR :", ASR[-1])
            print("test_acc:", test_accs, "     ASR:", ASR)

        matplotlib.style.use('seaborn')
        plt.switch_backend('agg')
        plt.figure()
        # plt.plot(train_ratio, train_accs, 'b-', lw=1)  # util.exp_moving_avg(train_accs, 0.85)
        plt.plot(test_ratio, test_accs, 'g:', lw=1)
        # plt.legend(['train', 'test'])
        # print(train_accs)
        # print(test_accs)
        # if test_dataset is not None:
        #     plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
        #     plt.legend(['train', 'val', 'test'])
        # else:
        #     plt.plot(best_val_epochs, best_val_accs, 'bo')
        #     plt.legend(['train', 'val'])
        # plt.savefig('results/iter/' + gen_prefix(args) + '_acc.png', dpi=600)
        # plt.close()

        # plt.figure()
        # # plt.plot(train_ratio, asr0_train, 'b-', lw=1)
        # plt.plot(test_ratio, asr0_test, 'g:', lw=1)
        # # plt.legend(['train', 'test'])
        # plt.savefig('results/iter/' + gen_prefix(args) + '_asr0.png', dpi=600)
        #
        # plt.figure()
        # # plt.plot(train_ratio, asr1_train, 'b-', lw=1)
        # plt.plot(test_ratio, asr1_test, 'g:', lw=1)
        # # plt.legend(['train', 'test'])
        # plt.savefig('results/iter/' + gen_prefix(args) + '_asr1.png', dpi=600)
        matplotlib.style.use('default')

        print(1)
