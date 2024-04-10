import torch


class JaccardLinkPrediction(torch.nn.Module):
    def __init__(self, num_nodes,emb_dim = 128):
        super(JaccardLinkPrediction, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, emb_dim)

    def forward(self, edge_index):
        src, dst = edge_index
        src_emb = self.embedding(src)
        dst_emb = self.embedding(dst)
        jaccard_sim = self.jaccard_similarity(src_emb, dst_emb)
        return torch.sigmoid(jaccard_sim)


    def jaccard_similarity(self, src_emb, dst_emb):
        intersection = torch.sum(src_emb * dst_emb, dim=1)
        union = torch.sum(src_emb + dst_emb > 0, dim=1)
        return intersection / union


