import torch
from sklearn.metrics import roc_auc_score, average_precision_score
def train(dataset, model, optimizer, epochs=1):
    for epoch in range(epochs):
        optimizer.zero_grad()

        pos_edge_label_index = dataset.edge_index
        pos_scores = model(pos_edge_label_index)

        num_nodes = dataset.num_nodes
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(0, num_nodes, (pos_edge_label_index.size(1),))
        ], dim=0)

        neg_scores = model(neg_edge_label_index)

        loss = torch.sigmoid(neg_scores - pos_scores).mean()
        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')



def evaluate(dataset, model):
    with torch.no_grad():
        pos_edge_label_index = dataset.edge_index

        pos_scores = model(pos_edge_label_index)

        num_nodes = dataset.num_nodes
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(0, num_nodes, (pos_edge_label_index.size(1),))
        ], dim=0)

        neg_scores = model(neg_edge_label_index)

        all_scores = torch.cat([pos_scores, neg_scores], dim=0)
        all_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)

        roc_auc = roc_auc_score(all_labels.cpu().numpy(), all_scores.cpu().numpy())
        avg_precision = average_precision_score(all_labels.cpu().numpy(), all_scores.cpu().numpy())

    return roc_auc, avg_precision