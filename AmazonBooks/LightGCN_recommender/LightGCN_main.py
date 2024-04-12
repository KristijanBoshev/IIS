from torch.optim import Adam
from torch_geometric.nn import LightGCN
from torch.utils.data import DataLoader
from torch_geometric.datasets import AmazonBook
from torch_geometric.transforms import RandomLinkSplit
from recommendation_lightgcn import train
from test_gcn import precision_at_k, recall_at_k, f1_at_k

if __name__ == '__main__':
    dataset = AmazonBook('../data')
    data = dataset[0]

    print(data)

    num_books, num_users = data['book'].num_nodes, data['user'].num_nodes
    num_nodes = data.num_nodes
    data = data.to_homogeneous()

    train_val_test_split = RandomLinkSplit(
                                           num_val= 0.1,
                                           num_test=0.2,
                                           add_negative_train_samples=True
                                           )

    train_data,val_data,test_data = train_val_test_split(data)

    train_loader = DataLoader(range(train_data.edge_index.size(1)),
                             shuffle=True,
                             batch_size=128)

    model = LightGCN(num_nodes=num_nodes, embedding_dim=128, num_layers=1)

    optimizer = Adam(model.parameters(), lr=0.01)

    train(dataset = data, train_loader = train_loader, model = model, optimizer = optimizer, num_users = num_users, num_books = num_books,epochs=1)

    for k in [1, 5, 10]:
        precision = precision_at_k(model, test_data, k)
        recall = recall_at_k(model, test_data, k)
        f1 = f1_at_k(model, test_data, k)
        print(f'Precision@{k}: {precision}, Recall@{k}: {recall}, F1@{k}: {f1}')





