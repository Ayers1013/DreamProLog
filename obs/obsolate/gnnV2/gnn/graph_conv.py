import tensorflow as tf
from gnn.tf_helpers import *

def gather_opt(params, indices): # if params = -1, uses zero tensor

    zeros = tf.zeros(tf.shape(params)[1:])
    zeros = tf.expand_dims(zeros, 0)
    params_ext = tf.concat([zeros, params], 0)
    return tf.gather(params_ext, indices+1)

class graph_start(tf.Module):
    def __init__(self, start_dims, graph):
        super().__init__()
        self.dim_nodes, self.dim_symbols, self.dim_clauses = start_dims

        self.node_emb=tf.Variable(tf.ones([4,self.dim_nodes]))
        self.symbol_emb=tf.zeros([2,self.dim_symbols])
        self.clause_emb=tf.Variable(tf.ones([6,self.dim_nodes]))
        self.graph=graph

    def __call__(self):
        nodes = tf.gather(self.node_emb, self.graph.ini_nodes)
        symbols = tf.gather(self.symbol_emb, self.graph.ini_symbols)
        clauses = tf.gather(self.clause_emb, self.graph.ini_clauses)

        return nodes, symbols, clauses 

# similar to segment_max, but being an odd function
def segment_minimax(data, segment_indices):
    a = tf.math.segment_max(data, segment_indices)
    b = tf.math.segment_min(data, segment_indices)
    return a+b

class graph_conv(tf.Module):
    def __init__(self, graph, output_dims = None, use_layer_norm = False):
        super().__init__()
        self.output_dims=output_dims
        self.use_layer_norm=use_layer_norm

        if use_layer_norm: self.layer=layer_norm_layer
        else: self.layer=concatenate_layer

        self.graph=graph

    def __call__(self, tensors):
        nodes, symbols, clauses = tensors
        if self.output_dims is None:
            out_dim_nodes = int(nodes.shape[-1])
            out_dim_symbols = int(symbols.shape[-1])
            out_dim_clauses = int(clauses.shape[-1])
        else:
            out_dim_nodes, out_dim_symbols, out_dim_clauses = self.output_dims

        in_nodes = []
        for n in self.graph.node_inputs:
            xn = gather_opt(nodes, n.nodes)
            dim = xn.shape[1]*xn.shape[2]
            xn = tf.reshape(xn, [-1, dim])
            xn = tf_linear_nob(xn, out_dim_nodes)
            xs = tf.gather(symbols, n.symbols) * tf.cast(tf.expand_dims(n.sgn, 1), dtype=tf.float32)
            xs = tf_linear(xs, out_dim_nodes)

            x = n.segments.collapse_nonzero(tf.nn.relu(xs + xn),
                                            [tf.math.segment_max, tf.math.segment_mean])
            x = n.segments.add_zeros(x)

            in_nodes.append(x)

        # out_nodes <- nodes + biases

        in_nodes.append(nodes)

        # out_nodes <- clauses

        nc = self.graph.node_c_inputs
        x = tf.gather(clauses, nc.data)
        x = nc.segments.collapse_nonzero(x, [tf.math.segment_max, tf.math.segment_mean])
        x = nc.segments.add_zeros(x)

        in_nodes.append(x)
        out_nodes = self.layer(in_nodes, out_dim_nodes)

        # out_symbols <- symbols, nodes

        sy = self.graph.symbol_inputs
        x = gather_opt(nodes, sy.nodes)
        dim = x.shape[1]*x.shape[2]
        x = tf.reshape(x, [-1, dim])
        x = tf_linear(x, out_dim_symbols)
        
        x = sy.segments.collapse(x*tf.cast(tf.expand_dims(sy.sgn,1), dtype=tf.float32),
                            operations = [tf.math.segment_mean, segment_minimax])

        
        #out_symbols = self.layer([x, symbols], out_dim_symbols,
        #                    activation_fn = tf.tanh, add_bias = False)

        # out_clauses <- nodes, clauses

        c = self.graph.clause_inputs
        x = c.segments.collapse(tf.gather(nodes, c.data),
                                [tf.math.segment_max, tf.math.segment_mean])

        #out_clauses = self.layer([x, clauses], out_dim_clauses)

        #out_symbols=symbols
        #out_clauses=clauses

        return out_nodes, out_symbols, out_clauses
