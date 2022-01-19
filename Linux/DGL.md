A DGL graph can store node features and edge features in two dictionary-like attributes called `ndata` and `edata`. In the DGL Cora dataset, the graph contains the following node features:

DGL 图可以将节点特征和边特征存储在两个类似字典的属性中，称为“ndata”和“edata”。在 DGL Cora 数据集中，该图包含以下节点特征：

- `train_mask`: A boolean tensor indicating whether the node is in the training set.

  指示节点是否在训练集中的布尔张量

- `val_mask`: A boolean tensor indicating whether the node is in the validation set.

  一个布尔张量，指示节点是否在验证集中。

- `test_mask`: A boolean tensor indicating whether the node is in the test set.

  一个布尔张量，指示节点是否在测试集中

- `label`: The ground truth node category.

- `feat`: The node features.



## Even more customization by user-defined function

DGL allows user-defined message and reduce function for the maximal expressiveness. Here is a user-defined message function that is equivalent to `fn.u_mul_e('h', 'w', 'm')`.

```
def u_mul_e_udf(edges):
    return {'m' : edges.src['h'] * edges.data['w']}
```

![Copy to clipboard](https://docs.dgl.ai/_static/copy-button.svg)

`edges` has three members: `src`, `data` and `dst`, representing the source node feature, edge feature, and destination node feature for all edges.

edge 有三个成员：src、data 和 dst，分别代表所有边的源节点特征、边特征和目的节点特征。



# Write your own GNN module

Sometimes, your model goes beyond simply stacking existing GNN modules. For example, you would like to invent a new way of aggregating neighbor information by considering node importance or edge weights.

By the end of this tutorial you will be able to

- Understand DGL’s message passing APIs.
- Implement GraphSAGE convolution module by your own.

This tutorial assumes that you already know [the basics of training a GNN for node classification](https://docs.dgl.ai/tutorials/blitz/1_introduction.html).

(Time estimate: 10 minutes)

```
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## Message passing and GNNs

DGL follows the *message passing paradigm* inspired by the Message Passing Neural Network proposed by [Gilmer et al.](https://arxiv.org/abs/1704.01212) Essentially, they found many GNN models can fit into the following framework:

$[m_{u\to v}^{(l)} = M^{(l)}\left(h_v^{(l-1)}, h_u^{(l-1)}, e_{u\to v}^{(l-1)}\right)]$

$[m_{v}^{(l)} = \sum_{u\in\mathcal{N}(v)}m_{u\to v}^{(l)}]$

$[h_v^{(l)} = U^{(l)}\left(h_v^{(l-1)}, m_v^{(l)}\right)]$

where DGL calls \($M^{(l)}$) the *message function*, \($\sum$\) the *reduce function* and \($U^{(l)}$\) the *update function*. Note that \($\sum$\) here can represent any function and is not necessarily a summation.

For example, the [GraphSAGE convolution (Hamilton et al., 2017)](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) takes the following mathematical form:

\[h_{\mathcal{N}(v)}^k\leftarrow \text{Average}\{h_u^{k-1},\forall u\in\mathcal{N}(v)\}\]

\[h_v^k\leftarrow \text{ReLU}\left(W^k\cdot \text{CONCAT}(h_v^{k-1}, h_{\mathcal{N}(v)}^k) \right)\]

You can see that message passing is directional: the message sent from one node \(u\) to other node \(v\) is not necessarily the same as the other message sent from node \(v\) to node \(u\) in the opposite direction.

Although DGL has builtin support of GraphSAGE via `dgl.nn.SAGEConv`, here is how you can implement GraphSAGE convolution in DGL by your own.

```
import dgl.function as fn

class SAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        """
        with g.local_scope():
            g.ndata['h'] = h
            # update_all is a message passing API.
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)
```

The central piece in this code is the [`g.update_all`](https://docs.dgl.ai/generated/dgl.DGLGraph.update_all.html#dgl.DGLGraph.update_all) function, which gathers and averages the neighbor features. There are three concepts here:

- Message function `fn.copy_u('h', 'm')` that copies the node feature under name `'h'` as *messages* sent to neighbors.
- Reduce function `fn.mean('m', 'h_N')` that averages all the received messages under name `'m'` and saves the result as a new node feature `'h_N'`.
- `update_all` tells DGL to trigger the message and reduce functions for all the nodes and edges.

Afterwards, you can stack your own GraphSAGE convolution layers to form a multi-layer GraphSAGE network.

```
class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
```

### Training loop

The following code for data loading and training loop is directly copied from the introduction tutorial.

```
import dgl.data

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    all_logits = []
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(200):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that we should only compute the losses of the nodes in the training set,
        # i.e. with train_mask 1.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_logits.append(logits.detach())

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

model = Model(g.ndata['feat'].shape[1], 16, dataset.num_classes)
train(g, model)
```

Out:

```
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
In epoch 0, loss: 1.949, val acc: 0.072 (best 0.072), test acc: 0.091 (best 0.091)
In epoch 5, loss: 1.872, val acc: 0.244 (best 0.244), test acc: 0.240 (best 0.240)
In epoch 10, loss: 1.722, val acc: 0.634 (best 0.634), test acc: 0.635 (best 0.635)
In epoch 15, loss: 1.489, val acc: 0.670 (best 0.670), test acc: 0.661 (best 0.661)
In epoch 20, loss: 1.185, val acc: 0.718 (best 0.718), test acc: 0.718 (best 0.718)
In epoch 25, loss: 0.850, val acc: 0.734 (best 0.734), test acc: 0.735 (best 0.735)
In epoch 30, loss: 0.547, val acc: 0.746 (best 0.746), test acc: 0.753 (best 0.753)
In epoch 35, loss: 0.323, val acc: 0.764 (best 0.764), test acc: 0.763 (best 0.762)
In epoch 40, loss: 0.183, val acc: 0.774 (best 0.774), test acc: 0.768 (best 0.768)
In epoch 45, loss: 0.105, val acc: 0.776 (best 0.776), test acc: 0.776 (best 0.775)
In epoch 50, loss: 0.063, val acc: 0.768 (best 0.776), test acc: 0.779 (best 0.775)
In epoch 55, loss: 0.040, val acc: 0.770 (best 0.776), test acc: 0.778 (best 0.775)
In epoch 60, loss: 0.028, val acc: 0.766 (best 0.776), test acc: 0.776 (best 0.775)
In epoch 65, loss: 0.021, val acc: 0.770 (best 0.776), test acc: 0.773 (best 0.775)
In epoch 70, loss: 0.016, val acc: 0.772 (best 0.776), test acc: 0.771 (best 0.775)
In epoch 75, loss: 0.013, val acc: 0.770 (best 0.776), test acc: 0.770 (best 0.775)
In epoch 80, loss: 0.011, val acc: 0.766 (best 0.776), test acc: 0.770 (best 0.775)
In epoch 85, loss: 0.010, val acc: 0.766 (best 0.776), test acc: 0.769 (best 0.775)
In epoch 90, loss: 0.009, val acc: 0.766 (best 0.776), test acc: 0.770 (best 0.775)
In epoch 95, loss: 0.008, val acc: 0.768 (best 0.776), test acc: 0.770 (best 0.775)
In epoch 100, loss: 0.007, val acc: 0.768 (best 0.776), test acc: 0.771 (best 0.775)
In epoch 105, loss: 0.007, val acc: 0.768 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 110, loss: 0.006, val acc: 0.768 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 115, loss: 0.006, val acc: 0.768 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 120, loss: 0.006, val acc: 0.768 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 125, loss: 0.005, val acc: 0.766 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 130, loss: 0.005, val acc: 0.764 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 135, loss: 0.005, val acc: 0.764 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 140, loss: 0.004, val acc: 0.764 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 145, loss: 0.004, val acc: 0.764 (best 0.776), test acc: 0.773 (best 0.775)
In epoch 150, loss: 0.004, val acc: 0.764 (best 0.776), test acc: 0.773 (best 0.775)
In epoch 155, loss: 0.004, val acc: 0.764 (best 0.776), test acc: 0.773 (best 0.775)
In epoch 160, loss: 0.004, val acc: 0.764 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 165, loss: 0.003, val acc: 0.764 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 170, loss: 0.003, val acc: 0.764 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 175, loss: 0.003, val acc: 0.762 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 180, loss: 0.003, val acc: 0.762 (best 0.776), test acc: 0.772 (best 0.775)
In epoch 185, loss: 0.003, val acc: 0.762 (best 0.776), test acc: 0.773 (best 0.775)
In epoch 190, loss: 0.003, val acc: 0.762 (best 0.776), test acc: 0.773 (best 0.775)
In epoch 195, loss: 0.003, val acc: 0.762 (best 0.776), test acc: 0.774 (best 0.775)
```

## More customization

In DGL, we provide many built-in message and reduce functions under the `dgl.function` package. You can find more details in [the API doc](https://docs.dgl.ai/api/python/dgl.function.html#apifunction).

These APIs allow one to quickly implement new graph convolution modules. For example, the following implements a new `SAGEConv` that aggregates neighbor representations using a weighted average. Note that `edata` member can hold edge features which can also take part in message passing.

```
class WeightedSAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model with edge weights.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super(WeightedSAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h, w):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        w : Tensor
            The edge weight.
        """
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['w'] = w
            g.update_all(message_func=fn.u_mul_e('h', 'w', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)
```

Because the graph in this dataset does not have edge weights, we manually assign all edge weights to one in the `forward()` function of the model. You can replace it with your own edge weights.

```
class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = WeightedSAGEConv(in_feats, h_feats)
        self.conv2 = WeightedSAGEConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat, torch.ones(g.num_edges()).to(g.device))
        h = F.relu(h)
        h = self.conv2(g, h, torch.ones(g.num_edges()).to(g.device))
        return h

model = Model(g.ndata['feat'].shape[1], 16, dataset.num_classes)
train(g, model)
```

Out:

```
In epoch 0, loss: 1.949, val acc: 0.162 (best 0.162), test acc: 0.144 (best 0.144)
In epoch 5, loss: 1.877, val acc: 0.348 (best 0.348), test acc: 0.344 (best 0.344)
In epoch 10, loss: 1.742, val acc: 0.376 (best 0.454), test acc: 0.352 (best 0.468)
In epoch 15, loss: 1.533, val acc: 0.428 (best 0.454), test acc: 0.414 (best 0.468)
In epoch 20, loss: 1.255, val acc: 0.574 (best 0.574), test acc: 0.582 (best 0.582)
In epoch 25, loss: 0.936, val acc: 0.656 (best 0.656), test acc: 0.684 (best 0.684)
In epoch 30, loss: 0.628, val acc: 0.700 (best 0.700), test acc: 0.725 (best 0.725)
In epoch 35, loss: 0.385, val acc: 0.716 (best 0.716), test acc: 0.743 (best 0.743)
In epoch 40, loss: 0.223, val acc: 0.722 (best 0.722), test acc: 0.755 (best 0.754)
In epoch 45, loss: 0.128, val acc: 0.736 (best 0.736), test acc: 0.763 (best 0.763)
In epoch 50, loss: 0.076, val acc: 0.750 (best 0.750), test acc: 0.764 (best 0.764)
In epoch 55, loss: 0.048, val acc: 0.750 (best 0.752), test acc: 0.762 (best 0.762)
In epoch 60, loss: 0.033, val acc: 0.754 (best 0.754), test acc: 0.764 (best 0.764)
In epoch 65, loss: 0.024, val acc: 0.754 (best 0.754), test acc: 0.764 (best 0.764)
In epoch 70, loss: 0.019, val acc: 0.752 (best 0.754), test acc: 0.763 (best 0.764)
In epoch 75, loss: 0.015, val acc: 0.746 (best 0.754), test acc: 0.764 (best 0.764)
In epoch 80, loss: 0.013, val acc: 0.746 (best 0.754), test acc: 0.763 (best 0.764)
In epoch 85, loss: 0.011, val acc: 0.744 (best 0.754), test acc: 0.764 (best 0.764)
In epoch 90, loss: 0.010, val acc: 0.744 (best 0.754), test acc: 0.764 (best 0.764)
In epoch 95, loss: 0.009, val acc: 0.746 (best 0.754), test acc: 0.762 (best 0.764)
In epoch 100, loss: 0.008, val acc: 0.748 (best 0.754), test acc: 0.761 (best 0.764)
In epoch 105, loss: 0.007, val acc: 0.748 (best 0.754), test acc: 0.759 (best 0.764)
In epoch 110, loss: 0.007, val acc: 0.748 (best 0.754), test acc: 0.759 (best 0.764)
In epoch 115, loss: 0.006, val acc: 0.748 (best 0.754), test acc: 0.759 (best 0.764)
In epoch 120, loss: 0.006, val acc: 0.748 (best 0.754), test acc: 0.759 (best 0.764)
In epoch 125, loss: 0.006, val acc: 0.748 (best 0.754), test acc: 0.759 (best 0.764)
In epoch 130, loss: 0.005, val acc: 0.748 (best 0.754), test acc: 0.760 (best 0.764)
In epoch 135, loss: 0.005, val acc: 0.748 (best 0.754), test acc: 0.760 (best 0.764)
In epoch 140, loss: 0.005, val acc: 0.748 (best 0.754), test acc: 0.760 (best 0.764)
In epoch 145, loss: 0.004, val acc: 0.748 (best 0.754), test acc: 0.760 (best 0.764)
In epoch 150, loss: 0.004, val acc: 0.750 (best 0.754), test acc: 0.760 (best 0.764)
In epoch 155, loss: 0.004, val acc: 0.748 (best 0.754), test acc: 0.758 (best 0.764)
In epoch 160, loss: 0.004, val acc: 0.750 (best 0.754), test acc: 0.757 (best 0.764)
In epoch 165, loss: 0.004, val acc: 0.750 (best 0.754), test acc: 0.757 (best 0.764)
In epoch 170, loss: 0.003, val acc: 0.750 (best 0.754), test acc: 0.757 (best 0.764)
In epoch 175, loss: 0.003, val acc: 0.750 (best 0.754), test acc: 0.757 (best 0.764)
In epoch 180, loss: 0.003, val acc: 0.750 (best 0.754), test acc: 0.757 (best 0.764)
In epoch 185, loss: 0.003, val acc: 0.754 (best 0.754), test acc: 0.758 (best 0.764)
In epoch 190, loss: 0.003, val acc: 0.754 (best 0.754), test acc: 0.758 (best 0.764)
In epoch 195, loss: 0.003, val acc: 0.754 (best 0.754), test acc: 0.758 (best 0.764)
```

## Even more customization by user-defined function

DGL allows user-defined message and reduce function for the maximal expressiveness. Here is a user-defined message function that is equivalent to `fn.u_mul_e('h', 'w', 'm')`.

```
def u_mul_e_udf(edges):
    return {'m' : edges.src['h'] * edges.data['w']}
```

`edges` has three members: `src`, `data` and `dst`, representing the source node feature, edge feature, and destination node feature for all edges.

You can also write your own reduce function. For example, the following is equivalent to the builtin `fn.sum('m', 'h')` function that sums up the incoming messages:

```
def sum_udf(nodes):
    return {'h': nodes.mailbox['m'].sum(1)}
```

In short, DGL will group the nodes by their in-degrees, and for each group DGL stacks the incoming messages along the second dimension. You can then perform a reduction along the second dimension to aggregate messages.

For more details on customizing message and reduce function with user-defined function, please refer to the [API reference](https://docs.dgl.ai/api/python/udf.html#apiudf).

## Best practice of writing custom GNN modules

DGL recommends the following practice ranked by preference:

- Use `dgl.nn` modules.
- Use `dgl.nn.functional` functions which contain lower-level complex operations such as computing a softmax for each node over incoming edges.
- Use `update_all` with builtin message and reduce functions.
- Use user-defined message or reduce functions.

## What’s next?

- [Writing Efficient Message Passing Code](https://docs.dgl.ai/guide/message-efficient.html#guide-message-passing-efficient).

```
# Thumbnail Courtesy: Representation Learning on Networks, Jure Leskovec, WWW 2018
# sphinx_gallery_thumbnail_path = '_static/blitz_3_message_passing.png'
```

**Total running time of the script:** ( 0 minutes  15.346 seconds)