import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from sklearn.metrics.pairwise import cosine_similarity
from utils.data_utils import dataset_iterator
from utils.data_prepro import UNK


def sigmoid_np(x, derivative=False):
    if derivative:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


def self_attention(inputs, return_alphas=False, project=True, reuse=None, name="self_attention"):
    with tf.variable_scope(name, reuse=reuse):

        hidden_size = inputs.shape[2].value

        if project:
            x = tf.layers.dense(inputs, units=hidden_size, use_bias=True, activation=tf.tanh)
        else:
            x = inputs

        weight = tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.01, seed=1227))

        x = tf.tensordot(x, weight, axes=1)

        alphas = tf.nn.softmax(x, axis=-1)

        output = tf.matmul(tf.transpose(inputs, [0, 2, 1]), alphas)
        output = tf.squeeze(output, -1)

        if not return_alphas:
            return output

        else:
            return output, alphas


def feed_forward_layer(inputs, hidden_units, output_units, use_bias=True, bias_init=0.0, activation=tf.tanh, reuse=None,
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


class ConceptualPrimitives:
    def __init__(self, cfg, verb_count, word_dict, verb_dict):
        self.cfg = cfg
        self.verb_count = verb_count
        self.word_dict = word_dict
        self.verb_dict = verb_dict
        self.rev_verb_dict = dict([(idx, verb) for verb, idx in self.verb_dict.items()])

        self._add_placeholders()
        self._build_model()

        print("total params: {}".format(self.count_params()), flush=True)
        self._init_session()

    def _init_session(self):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self.saver = tf.train.Saver(max_to_keep=10)

        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        print("restore model...", flush=True)
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)

        else:
            ckpt = tf.train.get_checkpoint_state(self.cfg.ckpt)  # get checkpoint state

        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored model from %s, done..." % ckpt.model_checkpoint_path, flush=True)

        else:
            raise ValueError("no pre-trained parameters in %s directory, please check or start a new session to train" %
                             self.cfg.ckpt)

    def save_session(self, global_step):
        self.saver.save(self.sess, self.cfg.ckpt + self.cfg.model_name,
                        global_step=global_step)

    def close_session(self):
        self.sess.close()

    @staticmethod
    def count_params(scope=None):
        if scope is None:
            return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

        else:
            return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope)])

    @staticmethod
    def get_shape(parameters):
        return parameters.get_shape().as_list()

    def get_feed_dict(self, data, training=False):
        feed_dict = {
            self.verb: data["verb"],
            self.left_context: data["l_context"],
            self.left_seq_len: data["l_seq_len"],
            self.right_context: data["r_context"],
            self.right_seq_len: data["r_seq_len"],
            self.training: training
        }

        return feed_dict

    def _add_placeholders(self):
        # add data placeholders
        self.left_context = tf.placeholder(name="left_context", shape=[None, None], dtype=tf.int32)
        self.left_seq_len = tf.placeholder(name="left_seq_len", shape=[None], dtype=tf.int32)
        self.right_context = tf.placeholder(name="right_context", shape=[None, None], dtype=tf.int32)
        self.right_seq_len = tf.placeholder(name="right_seq_len", shape=[None], dtype=tf.int32)
        self.verb = tf.placeholder(name="verb", shape=[None], dtype=tf.int32)

        # add hyper-parameter placeholders
        self.training = tf.placeholder(name="is_train", shape=[], dtype=tf.bool)

    def _build_model(self):
        with tf.device("/gpu:{}".format(self.cfg.gpu_idx[0])):
            with tf.variable_scope("word_lookup_table"):
                self.word_embs = tf.Variable(initial_value=np.load(self.cfg.word_vectors)["embeddings"],
                                             name="word_embeddings",
                                             dtype=tf.float32,
                                             trainable=self.cfg.tune_emb)

                unk = tf.get_variable(name="word_unk",
                                      shape=[1, self.cfg.word_dim],
                                      dtype=tf.float32,
                                      trainable=True)

                # the first embedding used for padding, zero initialization and fixed during training
                # the last embedding used for representing unknown word, randomly initialized and set as trainable
                self.word_embeddings = tf.concat([tf.zeros([1, self.cfg.word_dim]), self.word_embs, unk], axis=0)
                print("word embeddings shape: {}".format(self.get_shape(self.word_embeddings)), flush=True)

            with tf.variable_scope("verb_lookup_table"):
                self.verb_embs = tf.Variable(initial_value=np.load(self.cfg.verb_vectors)["embeddings"],
                                             name="verb_embeddings",
                                             dtype=tf.float32,
                                             trainable=self.cfg.tune_emb)

                unk = tf.get_variable(name="verb_unk",
                                      shape=[1, self.cfg.word_dim],
                                      dtype=tf.float32,
                                      trainable=True)

                # the last embedding used for representing unknown verb, randomly initialized and set as trainable
                self.verb_embeddings = tf.concat([self.verb_embs, unk], axis=0)
                print("verb embeddings shape: {}".format(self.get_shape(self.verb_embeddings)), flush=True)

            with tf.variable_scope("negative_sampling"):
                # cf. https://github.com/carpedm20/practice-tensorflow/blob/master/embedding/word2vec.py
                self.neg_verbs = []

                # negative sampling, used the same method in word2vec algorithm
                for i in range(self.cfg.neg_sample):
                    neg_verbs, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                        true_classes=tf.cast(tf.expand_dims(self.verb, axis=1), dtype=tf.int64),
                        num_true=1,
                        num_sampled=self.cfg.batch_size,
                        unique=False,
                        range_max=self.verb_embs.get_shape().as_list()[0],
                        distortion=0.75,
                        unigrams=self.verb_count,
                        seed=12345))

                    self.neg_verbs.append(neg_verbs)

                self.neg_verbs = tf.stack(self.neg_verbs, axis=1)
                print("negative samples shape: {}".format(self.get_shape(self.neg_verbs)), flush=True)

            with tf.variable_scope("embedding_lookup"):
                left_context_emb = tf.nn.embedding_lookup(self.word_embeddings, self.left_context)
                right_context_emb = tf.nn.embedding_lookup(self.word_embeddings, self.right_context)

                verb_emb = tf.nn.embedding_lookup(self.verb_embeddings, self.verb)
                neg_verb_emb = tf.nn.embedding_lookup(self.verb_embeddings, self.neg_verbs)

            # left context bidirectional lstm
            with tf.variable_scope("left_context_representation"):
                cell_fw = LSTMCell(num_units=self.cfg.num_units)
                cell_bw = LSTMCell(num_units=self.cfg.num_units)

                left_context_features, _ = bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                     cell_bw=cell_bw,
                                                                     inputs=left_context_emb,
                                                                     sequence_length=self.left_seq_len,
                                                                     dtype=tf.float32,
                                                                     time_major=False,
                                                                     scope="bidirectional_dynamic_rnn")

                left_context_features = tf.concat(left_context_features, axis=-1)

                # self-attention
                left_context_features = self_attention(left_context_features,
                                                       name="self_attention_left")

                left_weight = tf.get_variable(name="left_weight",
                                              shape=[2 * self.cfg.num_units, 2 * self.cfg.num_units],
                                              dtype=tf.float32)

                left_context_features = tf.tanh(tf.matmul(left_context_features, left_weight))
                print("left context shape: {}".format(self.get_shape(left_context_features)), flush=True)

            # right context bidirectional lstm
            with tf.variable_scope("right_context_representation"):
                cell_fw = LSTMCell(num_units=self.cfg.num_units)
                cell_bw = LSTMCell(num_units=self.cfg.num_units)

                right_context_features, _ = bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                      cell_bw=cell_bw,
                                                                      inputs=right_context_emb,
                                                                      sequence_length=self.right_seq_len,
                                                                      dtype=tf.float32,
                                                                      time_major=False,
                                                                      scope="bidirectional_dynamic_rnn")

                right_context_features = tf.concat(right_context_features, axis=-1)

                # self-attention
                right_context_features = self_attention(right_context_features,
                                                        name="self_attention_right")

                right_weight = tf.get_variable(name="right_weight",
                                               shape=[2 * self.cfg.num_units, 2 * self.cfg.num_units],
                                               dtype=tf.float32)

                right_context_features = tf.nn.tanh(tf.matmul(right_context_features, right_weight))
                print("right context shape: {}".format(self.get_shape(right_context_features)), flush=True)

        with tf.device("/gpu:{}".format(self.cfg.gpu_idx[1])):

            if self.cfg.use_ntn:
                with tf.variable_scope("neural_tensor_network"):
                    tensor = tf.get_variable(name="tensor",
                                             shape=[self.cfg.k,
                                                    2 * self.cfg.num_units,
                                                    2 * self.cfg.num_units],
                                             dtype=tf.float32)

                    weight = tf.get_variable(name="weight",
                                             shape=[4 * self.cfg.num_units,
                                                    self.cfg.k],
                                             dtype=tf.float32)

                    bias = tf.get_variable(name="bias",
                                           shape=[self.cfg.k],
                                           dtype=tf.float32)

                    # compute tensors
                    tensor_product = []
                    for idx in range(self.cfg.k):
                        tensor_product_ = tf.matmul(left_context_features, tensor[idx])
                        tensor_product_ = tensor_product_ * right_context_features
                        tensor_product_ = tf.reduce_sum(tensor_product_, axis=1)
                        tensor_product.append(tensor_product_)

                    tensor_product = tf.reshape(tf.concat(tensor_product, axis=0),
                                                shape=[-1, self.cfg.k])

                    features = tf.concat([left_context_features, right_context_features], axis=-1)
                    weight_product = tf.matmul(features, weight)

                    self.context = tf.tanh(tensor_product + weight_product + bias)

            else:
                with tf.variable_scope("context_fusion"):
                    raw_context = tf.concat([left_context_features, right_context_features], axis=-1)

                    self.context = feed_forward_layer(raw_context,
                                                      hidden_units=2 * self.cfg.k,
                                                      output_units=self.cfg.k,
                                                      use_bias=True,
                                                      bias_init=0.0,
                                                      activation=tf.tanh,
                                                      reuse=False,
                                                      name="context_projection")

            print("context representation shape: {}".format(self.get_shape(self.context)), flush=True)

            with tf.variable_scope("verb_representation"):
                self.target_verb = feed_forward_layer(inputs=verb_emb,
                                                      hidden_units=self.cfg.num_units,
                                                      output_units=self.cfg.k,
                                                      use_bias=True,
                                                      bias_init=0.0,
                                                      activation=tf.tanh,
                                                      reuse=tf.AUTO_REUSE,
                                                      name="feed_forward_layer")
                print("verb representation shape: {}".format(self.get_shape(self.target_verb)), flush=True)

                negative_verbs = feed_forward_layer(inputs=neg_verb_emb,
                                                    hidden_units=self.cfg.num_units,
                                                    output_units=self.cfg.k,
                                                    use_bias=True,
                                                    bias_init=0.0,
                                                    activation=tf.tanh,
                                                    reuse=tf.AUTO_REUSE,
                                                    name="feed_forward_layer")
                print("negative verb shape: {}".format(negative_verbs.get_shape().as_list()), flush=True)

            with tf.variable_scope("compute_logits"):
                expand_context = tf.tile(tf.expand_dims(self.context, axis=1),
                                         multiples=[1, self.cfg.neg_sample, 1])
                print("expanded context representation shape: {}".format(self.get_shape(expand_context)), flush=True)

                true_logits = tf.reduce_sum(tf.multiply(x=self.context,
                                                        y=self.target_verb), axis=1)
                print("true logits shape: {}".format(self.get_shape(true_logits)), flush=True)

                negative_logits = tf.reduce_sum(tf.multiply(x=expand_context,
                                                            y=negative_verbs), axis=2)
                print("negative logits shape: {}".format(self.get_shape(negative_logits)), flush=True)

            with tf.variable_scope("compute_loss"):
                self.true_prob = tf.nn.sigmoid(true_logits)
                true_loss = tf.log(tf.nn.sigmoid(self.true_prob))
                true_loss = -tf.reduce_mean(true_loss)

                # notice the minus sign for the negative logits
                negative_loss = tf.reduce_sum(tf.log(tf.nn.sigmoid(-negative_logits)), axis=1)
                negative_loss = -tf.reduce_mean(negative_loss)

                self.loss = true_loss + negative_loss

            global_step = tf.Variable(0, trainable=False, name='global_step')
            learning_rate = tf.train.exponential_decay(learning_rate=self.cfg.lr,
                                                       global_step=global_step,
                                                       decay_steps=int(1e5),
                                                       decay_rate=0.9994)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=global_step)

    def get_verb_embeddings(self, save_path=None):
        verb_embs = self.sess.run(self.verb_embs)  # do not contains UNK
        verb_dict = self.verb_dict.copy()
        verb_dict.pop(UNK)

        verb_vocab, verb_vectors = list(), list()
        for verb, idx in tqdm(verb_dict.items(), total=len(verb_dict), desc="extract verb embeddings"):
            verb_vocab.append(verb)
            verb_vectors.append(verb_embs[idx])

        result = {"vocab": verb_vocab, "vectors": np.asarray(verb_vectors)}

        if save_path is not None:
            with open(save_path, mode="wb") as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return result

    def get_verb_representation(self, save_path=None):
        verb_dict = self.verb_dict.copy()
        verb_dict.pop(UNK)  # remove UNK

        verb_vocab, verb_vectors = list(), list()
        for verb, idx in tqdm(verb_dict.items(), total=len(verb_dict), desc="extract verb representations"):
            verb_vector = self.sess.run(self.target_verb, feed_dict={self.verb: [idx]})
            verb_vector = np.reshape(verb_vector, newshape=(self.cfg.k,))
            verb_vocab.append(verb)
            verb_vectors.append(verb_vector)

        result = {"vocab": verb_vocab, "vectors": np.asarray(verb_vectors)}

        if save_path is not None:
            with open(save_path, mode="wb") as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return result

    def train(self, dataset):
        print("Start training...", flush=True)

        global_step = 0
        for epoch in range(1, self.cfg.epochs + 1):
            for i, data in enumerate(dataset_iterator(dataset, self.word_dict, self.verb_dict, self.cfg.batch_size)):

                global_step += 1
                feed_dict = self.get_feed_dict(data, training=True)
                _, losses = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                if (i + 1) % 5000 == 0:
                    train_loss = "{0:.4f}".format(losses)
                    print("Epoch: {} / {}, Local Steps: {}, Global Steps: {}, Training Loss: {}".format(
                        epoch, self.cfg.epochs, i + 1, global_step, train_loss), flush=True)

                if global_step % 10000 == 0:
                    self.save_session(global_step)

            # save the model
            self.save_session(global_step)

        print("Training process finished...", flush=True)

    def infer(self, sentence, verb, top_n=10, method="multiply"):
        # pre-process inputs
        verb = verb.lower()
        words = word_tokenize(sentence.lower(), language="english")
        index = words.index(verb)

        # build feed data
        l_context = [self.word_dict[w] if w in self.word_dict else self.word_dict[UNK] for w in words[0:index]]
        l_seq_len = len(l_context)
        r_context = [self.word_dict[w] if w in self.word_dict else self.word_dict[UNK] for w in words[index + 1:]]
        r_seq_len = len(r_context)
        verb = self.verb_dict[verb] if verb in self.verb_dict else self.verb_dict[UNK]

        # deal with substitution of target verb
        candidate_verbs = [x for x in list(self.verb_dict.values())]
        candidate_verbs.remove(self.verb_dict[UNK])
        candidate_verbs.remove(verb)

        # compute context, target verb and candidates representations in the embedding hyperspace
        context_vec, verb_vec = self.sess.run([self.context, self.target_verb],
                                              feed_dict={
                                                  self.left_context: [l_context],
                                                  self.left_seq_len: [l_seq_len],
                                                  self.right_context: [r_context],
                                                  self.right_seq_len: [r_seq_len],
                                                  self.verb: [verb]
                                              })

        candidate_vecs = self.sess.run(self.target_verb, feed_dict={self.verb: candidate_verbs})

        verb_candidate_similarity = cosine_similarity(verb_vec, candidate_vecs)
        context_candidate_similarity = cosine_similarity(context_vec, candidate_vecs)

        if method == "multiply":
            similarities = verb_candidate_similarity * context_candidate_similarity
        elif method == "add":
            similarities = verb_candidate_similarity + context_candidate_similarity
        else:
            raise ValueError("Unsupported similarity method...")
        similarities = np.reshape(similarities, newshape=(similarities.shape[1],))

        candidate_dict = dict()
        for i in range(similarities.shape[0]):
            candidate_dict[candidate_verbs[i]] = similarities[i]

        top_candidates = sorted(candidate_dict.items(), key=lambda kv: kv[1], reverse=True)[0:top_n]
        top_candidates = [self.rev_verb_dict[x] for x, _ in top_candidates]

        return top_candidates
