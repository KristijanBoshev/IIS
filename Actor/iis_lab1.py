# -*- coding: utf-8 -*-
"""IIS_Lab1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/KristijanBoshev/IIS/blob/main/Actor/IIS_Lab1.ipynb
"""

!pip install torch_geometric

pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

from torch_geometric.datasets import Actor
from models import SAGEConv, GCNConv
from model_utils import train
from torch_geometric.nn import Node2Vec
from node_embeddings import train_Node

data = Actor('../data')
df = data[0]

df

model_Node = Node2Vec(df.edge_index,
                    embedding_dim=50,
                     walk_length=30,
                     context_size=10,
                     walks_per_node=20,
                     num_negative_samples=1,
                     p=200, q=1,
                     sparse=True)

train_Node(model_Node)

labels = df.y.detach().cpu().numpy()
node_embeddings = model_Node().detach().cpu().numpy()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(node_embeddings, labels,
                                                        test_size=0.1,
                                                        stratify=labels)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(train_x, train_y)

preds = classifier.predict(test_x)

from sklearn.metrics import accuracy_score
print(f'Accuracy: {accuracy_score(preds, test_y)}')

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
node_embeddings_2d = tsne.fit_transform(node_embeddings)

import matplotlib.pyplot as plt

plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1],
                c=labels, cmap='jet', alpha=0.7)

model_GNN = GCNConv(-1,10)

from torch.optim import Adam
from torch.nn import CrossEntropyLoss

optimizer = Adam(model_GNN.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

node_embeddings_pre = model_GNN(df.x, df.edge_index)

node_embeddings_pre_detached = node_embeddings_pre.detach().numpy()

tsne = TSNE(n_components=2)
node_embeddings_2d_pre = tsne.fit_transform(node_embeddings_pre_detached)

plt.scatter(node_embeddings_2d_pre[:, 0], node_embeddings_2d_pre[:, 1],
                c=labels, cmap='jet', alpha=0.7)

train(model_GNN, df, optimizer, criterion, 10)

node_embeddings_after = model_GNN(df.x, df.edge_index)

node_embeddings_after_detached = node_embeddings_pre.detach().numpy()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(node_embeddings_after_detached, labels,
                                                        test_size=0.1,
                                                        stratify=labels)

classifier_GNN = RandomForestClassifier(n_estimators=50)
classifier_GNN.fit(train_x, train_y)

preds_GNN = classifier_GNN.predict(test_x)

print(f"Accuracy score: {accuracy_score(preds_GNN,test_y)}")

tsne = TSNE(n_components=2)
node_embeddings_2d_after = tsne.fit_transform(node_embeddings_after_detached)

plt.scatter(node_embeddings_2d_pre[:, 0], node_embeddings_2d_pre[:, 1],
                c=labels, cmap='jet', alpha=0.7)