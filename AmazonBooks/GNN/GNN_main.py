from torch.optim import SGD
from torch_geometric.datasets import AmazonBook
from torch_geometric.transforms import RandomLinkSplit
from gnn_link_prediction import Model, train_link_prediction

if __name__ == '__main__':
    dataset = AmazonBook('../data/AmazonBook')
    data = dataset[0]

    train_val_test_split = RandomLinkSplit(num_val=0.2,
                                           num_test=0.2,
                                           add_negative_train_samples=True,
                                           edge_types=('user', 'rates', 'book'),
                                           rev_edge_types=('book', 'rated_by', 'user'))

    train_data, val_data, test_data = train_val_test_split(data)

    model = Model(hidden_channels=128, data=data)

    optimizer = SGD(model.parameters(), lr=0.01, momentum=0)
    train_link_prediction(model = model, val_data = val_data, train_data = train_data, optimizer = optimizer)

