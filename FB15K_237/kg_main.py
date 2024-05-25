from torch.optim import Adam
from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import TransE, ComplEx
from kg_model_utils import train, evaluate

if __name__ == '__main__':
    train_data = FB15k_237('../data/FB15k', split='train')[0]
    val_data = FB15k_237('../data/FB15k', split='val')[0]
    test_data = FB15k_237('../data/FB15k', split='test')[0]

    model = TransE(num_nodes=train_data.num_nodes,
                   num_relations=train_data.num_edge_types,
                   hidden_channels=50)

    model2 = ComplEx(num_nodes=train_data.num_nodes,
                     num_relations=train_data.num_edge_types,
                     hidden_channels=50)

    loader = model.loader(head_index=train_data.edge_index[0],
                          rel_type=train_data.edge_type,
                          tail_index=train_data.edge_index[1],
                          batch_size=1000,
                          shuffle=True)

    optimizer = Adam(model.parameters(), lr=0.01)

    optimizer2 = Adam(model2.parameters, lr=0.01)

    train = train(model, loader, optimizer)

    train2 = train(model2, loader, optimizer2)

    rank, mrr, hits10 = model.test(head_index=test_data.edge_index[0],
                                   rel_type=test_data.edge_type,
                                   tail_index=test_data.edge_index[1],
                                   batch_size=1000, k=10)

    eval = evaluate(model, loader)

    rank, mrr, hits10 = model2.test(head_index=test_data.edge_index[0],
                                    rel_type=test_data.edge_type,
                                    tail_index=test_data.edge_index[1],
                                    batch_size=1000, k=10)

    eval2 = evaluate(model2, loader)

    print(eval)
    print(eval2)

