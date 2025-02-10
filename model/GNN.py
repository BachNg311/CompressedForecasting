import torch

import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1)**0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GCN_for_TimeSeries(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_for_TimeSeries, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # x shape: (batch_size, num_nodes, features) - > (batch_size, num_nodes, features)
        # adj shape: (num_nodes, num_nodes)

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x # No softmax,  regression task

if __name__ == '__main__':
    # Example usage
    # Assuming you have time series data represented as a graph
    # where each node is a time series and edges represent relationships
    # between them.

    # Parameters
    batch_size = 32
    num_nodes = 10  # Number of time series
    num_features = 64 # Features for each time series (e.g., past values, embeddings)
    hidden_dim = 32
    output_dim = 1 # Predicting one step ahead
    dropout = 0.5

    # Generate synthetic data
    x = torch.randn(batch_size, num_nodes, num_features)
    adj = torch.randn(num_nodes, num_nodes) #Adjacency matrix (replace with your actual adjacency matrix)
    adj = (adj + adj.T)/2
    adj = torch.sigmoid(adj) # Example: Using sigmoid to get edge weights between 0 and 1

    # Instantiate the model
    model = GCN_for_TimeSeries(num_features, hidden_dim, output_dim, dropout)

    # Forward pass
    output = model(x, adj)

    # Output shape: (batch_size, num_nodes, output_dim)
    print(output.shape)

    # Loss and optimization 
    target = torch.randn(batch_size, num_nodes, output_dim)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())