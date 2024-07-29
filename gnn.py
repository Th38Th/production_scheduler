import builtins
import collections

import tensorflow as tf
import spektral as sp
from scipy import sparse
import numpy as np
from spektral.utils.keras import deserialize_kwarg, is_keras_kwarg
from spektral.models.general_gnn import MLP


class SimpleMaskingLayer(tf.keras.layers.Layer):

    def __init__(self, mask):
        super(SimpleMaskingLayer, self).__init__()
        self.mask = mask

    def call(self, inputs):
        return inputs * self.mask


class FullMessagePassing(sp.layers.MessagePassing):
    r"""
        An implementation of a Graph Network (GN) Block from the paper

        > [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261) <br>
        > Peter W. Battaglia et al.

         **Mode**: single, disjoint.

        **This layer and all of its extensions expect a sparse adjacency matrix.**

    """

    def __init__(self,
                 n_out_nodes,
                 n_out_edges,
                 n_out_global,
                 aggregate='mean',
                 node_aggregation='sum',
                 edge_aggregation='sum',
                 update_nodes=None,
                 update_edges=None,
                 update_global=None,
                 **kwargs):
        super().__init__(aggregate=aggregate, **kwargs)

        self.update_nodes = update_nodes or MLP(n_out_nodes)
        self.update_edges = update_edges or MLP(n_out_edges)
        self.update_global = update_global or MLP(n_out_global)

        self.node_aggregation = node_aggregation \
            if isinstance(node_aggregation, collections.abc.Callable) \
            else getattr(tf.math, 'reduce_' + node_aggregation, tf.math.reduce_sum)
        self.edge_aggregation = edge_aggregation \
            if isinstance(edge_aggregation, collections.abc.Callable) \
            else getattr(tf.math, 'reduce_' + edge_aggregation, tf.math.reduce_sum)

    def call(self, inputs, **kwargs):
        x, a, e, u = inputs
        return self.propagate(x, a, e, u)

    def propagate(self, x, a, e, u, **kwargs):
        self.n_nodes = tf.shape(x)[-2]
        self.n_edges = tf.shape(e)[-2]

        self.index_sources = a.indices[:, 0]
        self.index_targets = a.indices[:, 1]

        # Update each edge based on its features,
        # the features of the nodes it links,
        # and the global features of the graph
        edge_messages = self.message(x, e)
        u_vector_e = tf.repeat(u, self.n_edges, axis=0)
        update_args = tf.concat([edge_messages, u_vector_e], axis=1)
        e = self.update_edges(update_args)

        # Update each node based on
        # the updated edge features,
        # its own features,
        # the features of its neighbors
        # and the global features of the graph
        node_messages = self.message(x, e)
        embeddings = self.aggregate(node_messages)
        u_vector_v = tf.repeat(u, self.n_nodes, axis=0)
        update_args = tf.concat([embeddings, u_vector_v], axis=1)
        x = self.update_nodes(update_args)

        # Aggregate edges and nodes and compute global update
        x_ = self.node_aggregation(x, keepdims=True, axis=0)
        e_ = self.edge_aggregation(e, keepdims=True, axis=0)
        update_args = tf.concat([x_, e_, u], axis=1)
        u = self.update_global(update_args)

        return x, a, e, u

    def message(self, x, e, **kwargs):
        return tf.concat(
            [
                e,
                self.get_sources(x),
                self.get_targets(x)]
            ,
            axis=1
        )

    def aggregate(self, messages, **kwargs):
        return self.agg(messages, self.index_sources, self.n_nodes)


class GraphMLP(tf.keras.Model):
    def __init__(self,
                 node_output,
                 edge_output,
                 global_output,
                 node_hidden=256,
                 node_layers=2,
                 node_batch_norm=True,
                 node_dropout=0.0,
                 node_activation='relu',
                 node_final_activation='relu',
                 edge_hidden=256,
                 edge_layers=2,
                 edge_batch_norm=True,
                 edge_dropout=0.0,
                 edge_activation='relu',
                 edge_final_activation='relu',
                 global_hidden=256,
                 global_layers=2,
                 global_batch_norm=True,
                 global_dropout=0.0,
                 global_activation='relu',
                 global_final_activation='relu'):
        super().__init__()

        self.node_mlp = MLP(node_output, node_hidden, node_layers, node_batch_norm, node_dropout, node_activation,
                            node_final_activation)
        self.edge_mlp = MLP(edge_output, edge_hidden, edge_layers, edge_batch_norm, edge_dropout, edge_activation,
                            edge_final_activation)
        self.global_mlp = MLP(global_output, global_hidden, global_layers, global_batch_norm, global_dropout,
                              global_activation,
                              global_final_activation)

    def call(self, inputs):
        x, a, e, u = inputs
        x = self.node_mlp(x)
        e = self.edge_mlp(e)
        u = self.global_mlp(u)
        return x, a, e, u


class PMSPGNN(tf.keras.Model):
    def __init__(self,
                 node_output,
                 edge_output,
                 global_output,
                 graph_network=2,
                 aggregate='mean',
                 update=(3, 1),
                 global_hidden=256,
                 global_embedding=2,
                 global_dropout=0.0,
                 global_batch_norm=True,
                 global_hidden_activation='leaky_relu',
                 global_activation='softmax',
                 node_hidden=256,
                 node_embedding=2,
                 node_dropout=0.0,
                 node_batch_norm=True,
                 node_aggregate="sum",
                 node_hidden_activation='leaky_relu',
                 node_activation='softmax',
                 edge_hidden=256,
                 edge_embedding=2,
                 edge_dropout=0.0,
                 edge_batch_norm=True,
                 edge_aggregate="sum",
                 edge_hidden_activation='leaky_relu',
                 edge_activation='softmax',
                 ):
        super().__init__()

        self.embedding = GraphMLP(
            node_hidden,
            edge_hidden,
            global_hidden,
            node_hidden,
            node_embedding,
            node_batch_norm,
            node_dropout,
            node_hidden_activation,
            node_hidden_activation,
            edge_hidden,
            edge_embedding,
            edge_batch_norm,
            edge_dropout,
            edge_hidden_activation,
            edge_hidden_activation,
            global_hidden,
            global_embedding,
            global_batch_norm,
            global_dropout,
            global_hidden_activation,
            global_hidden_activation,
        )

        self.gnn = [
            FullMessagePassing(
                node_output, edge_output, global_output,
                aggregate=aggregate,
                node_aggregate=node_aggregate,
                edge_aggregate=edge_aggregate,
                update_nodes=MLP(node_output, layers=update[i], activation=node_hidden_activation),
                update_edges=MLP(edge_output, layers=update[i], activation=edge_hidden_activation),
                update_global=MLP(global_output, layers=update[i], activation=global_hidden_activation),
            ) for i in range(graph_network)]

        self.node_activation = tf.keras.layers.Activation(node_activation)
        self.edge_activation = tf.keras.layers.Activation(edge_activation)
        self.global_activation = tf.keras.layers.Activation(global_activation)

    def call(self, inputs):
        x, a, e = inputs
        u = tf.zeros((1, 1))
        inputs = (x, a, e, u)
        out = self.embedding(inputs)
        for layer in self.gnn:
            out = layer(out)
        x, a, e, u = out
        x = self.node_activation(x)
        e = self.edge_activation(e)
        u = self.global_activation(u)
        # We only care about the edges
        # From this point onwards
        # But also the no-op indicator
        # Which will be the global attributes
        out = e
        return out
