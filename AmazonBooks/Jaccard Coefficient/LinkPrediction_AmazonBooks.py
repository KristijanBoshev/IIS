import torch
from torch_geometric.datasets import AmazonBook
from torch_geometric.transforms import RandomLinkSplit
from JaccardModel import JaccardLinkPrediction
from Jac_Train import train, evaluate


if __name__ == '__main__':
    dataset = AmazonBook('../data')
    data = dataset[0]
    data = data.to_homogeneous()

    train_val_test_split = RandomLinkSplit(num_val=0.2,
                                           num_test=0.2,
                                           add_negative_train_samples=True,
                                           edge_types=('book', 'rated_by', 'user'))


    train_data, val_data, test_data = train_val_test_split(data)


    model = JaccardLinkPrediction(num_nodes=data.num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train(dataset=data, model=model, optimizer=optimizer)

    roc_auc, avg_precision = evaluate(dataset, model)

    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'Average Precision: {avg_precision:.4f}')













