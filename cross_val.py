import networkx as nx
import numpy as np
import torch

import pickle
import random

from graph_sampler import GraphSampler


def prepare_val_data(graphs, args, val_idx, max_nodes=0):
    random.shuffle(graphs)
    # graphs = graphs[0:1100]
    val_size = len(graphs) // 10
    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx + 1) * val_size:]
    val_graphs = graphs[val_idx * val_size: (val_idx + 1) * val_size]

    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
          max([G.number_of_nodes() for G in graphs]), ', '
                                                      "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
          ', '
          "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=args.normalize, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False)
    # ,
    # num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=args.normalize, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False)
    # ,
    # num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, \
           dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim


def prepare_val_test_data(graphs, args, val_idx, max_nodes=0):
    sub_list = []
    for sub in graphs:
        print(sub.graph['label'])
        if sub.graph['label'] == 1:
            sub_list.append(sub)
    # graphs = graphs[0:7000] +sub_list
    random.shuffle(graphs)
    # graphs = graphs[0:10000]
    val_size = len(graphs) // 10  #FB
    train_graphs = graphs[:val_idx * val_size]


    if val_idx < 9:
        test_idx = val_idx + 1
    else:
        test_idx = 0
    if val_idx < 8:
        train_graphs = train_graphs + graphs[(val_idx + 2) * val_size:]

    print('len',len(sub_list))
    # train_graphs = train_graphs +sub_list*10
    # random.shuffle(train_graphs)

    val_graphs = graphs[val_idx * val_size: (val_idx + 1) * val_size]
    test_graphs = graphs[test_idx * val_size: (test_idx + 1) * val_size]

    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs),
          '; Num test graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
          max([G.number_of_nodes() for G in graphs]), ', '
                                                      "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
          ', '
          "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=args.normalize, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True)
    # ,
    # num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=args.normalize, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True)
    # ,
    # num_workers=args.num_workers)
    dataset_sampler = GraphSampler(test_graphs, normalize=args.normalize, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
           dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim