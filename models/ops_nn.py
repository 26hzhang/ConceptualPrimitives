import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn


def embedding_lookup(inputs1, inputs2, word_dim, vectors, tune_emb, use_pad=True, name="embedding_lookup"):
    with tf.variable_scope(name):
        embs = tf.Variable(initial_value=np.load(vectors)["embeddings"],
                           name="embeddings",
                           dtype=tf.float32,
                           trainable=tune_emb)

        unk = tf.get_variable(name="unk",
                              shape=[1, word_dim],
                              dtype=tf.float32,
                              trainable=True)
        if use_pad:
            embeddings = tf.concat([tf.zeros([1, word_dim]), unk, embs], axis=0)
        else:
            embeddings = tf.concat([unk, embs], axis=0)

        inputs1_emb = tf.nn.embedding_lookup(embeddings, inputs1)
        inputs2_emb = tf.nn.embedding_lookup(embeddings, inputs2)

        return inputs1_emb, inputs2_emb, embeddings


def self_attention(inputs, num_units, return_alphas=False, reuse=None, name="self_attention"):
    with tf.variable_scope(name, reuse=reuse):

        x_proj = tf.layers.dense(inputs,
                                 units=num_units,
                                 use_bias=True,
                                 activation=tf.nn.tanh,
                                 name="projection")  # batch_sz x seq_len x num_units

        weight = tf.get_variable(name="weight",
                                 shape=[num_units, 1],
                                 initializer=tf.random_normal_initializer(stddev=0.01,
                                                                          seed=12345),
                                 trainable=True)  # num_units x 1

        x = tf.tensordot(x_proj, weight, axes=1)  # batch_sz x seq_len x 1

        # softmax should be applied to the seq_len dimension to compute the importance of each token in the sequence
        alphas = tf.nn.softmax(x, axis=1)

        output = tf.matmul(tf.transpose(inputs, [0, 2, 1]), alphas)  # batch_sz x inputs_last_dim x 1
        output = tf.squeeze(output, -1)

        if not return_alphas:
            return output

        else:
            return output, alphas


def feed_forward_nets(inputs, hidden_units, output_units, use_bias=True, bias_init=0.0, activation=tf.tanh, reuse=None,
                      name="feed_forward_layer"):
    with tf.variable_scope(name, reuse=reuse):

        # hidden layer
        hidden_output = tf.layers.dense(inputs,
                                        units=hidden_units,
                                        use_bias=use_bias,
                                        activation=activation,
                                        bias_initializer=tf.constant_initializer(bias_init),
                                        name="hidden_dense")

        # output layer
        output = tf.layers.dense(hidden_output,
                                 units=output_units,
                                 use_bias=use_bias,
                                 activation=activation,
                                 bias_initializer=tf.constant_initializer(bias_init),
                                 name="output_dense")
        return output


def context_representation(inputs, seq_len, num_units, activation=tf.nn.tanh, use_bias=False, reuse=None,
                           name="context_rep"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        cell_fw = LSTMCell(num_units=num_units)
        cell_bw = LSTMCell(num_units=num_units)

        context_features, _ = bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                        cell_bw=cell_bw,
                                                        inputs=inputs,
                                                        sequence_length=seq_len,
                                                        dtype=tf.float32,
                                                        time_major=False,
                                                        scope="bidirectional_dynamic_rnn")

        context_features = tf.concat(context_features, axis=-1)

        # self-attention
        context_features = self_attention(context_features,
                                          num_units=num_units,
                                          return_alphas=False,
                                          reuse=reuse,
                                          name="self_attention")

        # dense layer project
        context_features = tf.layers.dense(context_features,
                                           units=2 * num_units,
                                           use_bias=use_bias,
                                           kernel_initializer=tf.glorot_uniform_initializer(),
                                           activation=activation,
                                           name="context_project")

        return context_features


def neural_tensor_net(inputs1, inputs2, hidden_units, output_units, activation=tf.nn.tanh, reuse=None,
                      name="neural_tensor_network"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        tensor = tf.get_variable(name="tensor",
                                 shape=[output_units, hidden_units, hidden_units],
                                 initializer=tf.glorot_uniform_initializer(),
                                 dtype=tf.float32)

        weight = tf.get_variable(name="weight",
                                 shape=[2 * hidden_units, output_units],
                                 initializer=tf.glorot_uniform_initializer(),
                                 dtype=tf.float32)

        bias = tf.get_variable(name="bias",
                               shape=[output_units],
                               initializer=tf.constant_initializer(0.0),
                               dtype=tf.float32)

        # compute tensors
        tensor_product = []
        for idx in range(output_units):
            tensor_product_ = tf.matmul(inputs1, tensor[idx])
            tensor_product_ = tensor_product_ * inputs2
            tensor_product_ = tf.reduce_sum(tensor_product_, axis=1)
            tensor_product.append(tensor_product_)

        tensor_product = tf.reshape(tf.concat(tensor_product, axis=0),
                                    shape=[-1, output_units])

        features = tf.concat([inputs1, inputs2], axis=-1)
        weight_product = tf.matmul(features, weight)

        outputs = activation(tensor_product + weight_product + bias)

        return outputs


def naive_fusion(inputs1, inputs2, output_units, use_bias=True, bias_init=0.0, activation=tf.nn.tanh, reuse=None,
                 name="naive_fusion"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        inputs = tf.concat([inputs1, inputs2], axis=-1)

        outputs = tf.layers.dense(inputs,
                                  units=output_units,
                                  use_bias=use_bias,
                                  bias_initializer=tf.constant_initializer(bias_init),
                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                  activation=activation,
                                  name="project")

        return outputs
