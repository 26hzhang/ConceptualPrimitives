import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from sklearn.metrics.pairwise import cosine_similarity
from data_fns import dataset_iterator, convert_single, convert_batch, UNK
from optimization import create_optimizer


def self_attention(inputs, num_units, init_range, activation=tf.tanh, reuse=None, name="self_attention"):
    with tf.variable_scope(name, reuse=reuse):

        x_proj = tf.layers.dense(inputs,
                                 units=num_units,
                                 use_bias=True,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=init_range),
                                 bias_initializer=tf.zeros_initializer(),
                                 activation=activation,
                                 name="projection")  # batch_sz x seq_len x num_units

        weight = tf.get_variable(name="weight",
                                 shape=[num_units, 1],
                                 initializer=tf.truncated_normal_initializer(stddev=init_range),
                                 trainable=True)  # num_units x 1

        x = tf.tensordot(x_proj, weight, axes=1)  # batch_sz x seq_len x 1

        # softmax should be applied to the seq_len dimension to compute the importance of each token in the sequence
        alphas = tf.nn.softmax(x, axis=1)

        output = tf.matmul(tf.transpose(inputs, [0, 2, 1]), alphas)  # batch_sz x inputs_last_dim x 1
        output = tf.squeeze(output, -1)

        return output


def feed_forward_layer(inputs, hidden_units, output_units, init_range, use_bias=True, activation=tf.tanh, reuse=None,
                       name="feed_forward_layer"):
    with tf.variable_scope(name, reuse=reuse):
        # hidden layer
        hidden_output = tf.layers.dense(inputs,
                                        units=hidden_units,
                                        use_bias=use_bias,
                                        activation=activation,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=init_range),
                                        bias_initializer=tf.zeros_initializer(),
                                        name="hidden_dense")

        # output layer
        output = tf.layers.dense(hidden_output,
                                 units=output_units,
                                 use_bias=use_bias,
                                 activation=activation,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=init_range),
                                 bias_initializer=tf.zeros_initializer(),
                                 name="output_dense")
        return output


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.

    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


class ConceptualPrimitives:
    def __init__(self, config, verb_count, word_dict, verb_dict, num_train_steps):
        # tf.random.set_random_seed(config.random_seed)
        self.config = config
        self.num_train_steps = num_train_steps
        self.verb_count = verb_count
        self.word_dict = word_dict
        self.word_dict_size = len(self.word_dict)
        self.verb_dict = verb_dict
        self.verb_dict_size = len(self.verb_dict)
        self.rev_verb_dict = dict([(idx, verb) for verb, idx in self.verb_dict.items()])

        self.activation = gelu

        self._add_placeholders()
        self._build_model()

        if self.config.mode == "train":
            print("total params: {}".format(self.count_params()), flush=True)
        self._init_session()

    def _init_session(self):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self.saver = tf.train.Saver(max_to_keep=3)

        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)

        else:
            ckpt = tf.train.get_checkpoint_state(self.config.ckpt)  # get checkpoint state

        if ckpt and ckpt.model_checkpoint_path:  # restore session
            print("restoring from latest checkpoints \"%s\"" % ckpt.model_checkpoint_path, end=", ", flush=True)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("done...", flush=True)

        else:
            raise ValueError("no pretrained parameters in %s directory, please check or start a new session to train" %
                             self.config.ckpt)

    def save_session(self, global_step):
        self.saver.save(sess=self.sess,
                        save_path=self.config.ckpt + self.config.model_name,
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
        self.training = tf.placeholder(name="training", shape=[], dtype=tf.bool)

    def _build_model(self):
        with tf.variable_scope("word_lookup_table"):
            self.word_embs = tf.Variable(initial_value=np.load(self.config.word_vectors)["embeddings"],
                                         name="word_embeddings",
                                         dtype=tf.float32,
                                         trainable=self.config.tune_emb)

            unk = tf.get_variable(name="word_unk",
                                  shape=[1, self.config.word_dim],
                                  initializer=tf.truncated_normal_initializer(stddev=self.config.initializer_range),
                                  dtype=tf.float32,
                                  trainable=True)

            # the first embedding used for padding, zero initialization and fixed during training
            # the last embedding used for representing unknown word, randomly initialized and trainable
            self.word_embeddings = tf.concat([tf.zeros([1, self.config.word_dim]), unk, self.word_embs], axis=0)

            if self.config.mode == "train":
                print("word embeddings shape: {}".format(self.get_shape(self.word_embeddings)), flush=True)

        with tf.variable_scope("verb_lookup_table"):
            self.verb_embs = tf.Variable(initial_value=np.load(self.config.verb_vectors)["embeddings"],
                                         name="verb_embeddings",
                                         dtype=tf.float32,
                                         trainable=self.config.tune_emb)

            unk = tf.get_variable(name="verb_unk",
                                  shape=[1, self.config.word_dim],
                                  initializer=tf.truncated_normal_initializer(stddev=self.config.initializer_range),
                                  dtype=tf.float32,
                                  trainable=True)

            # the last embedding used for representing unknown verb, randomly initialized and set as trainable
            self.verb_embeddings = tf.concat([unk, self.verb_embs], axis=0)
            self.verb_bias = tf.Variable(tf.zeros([self.verb_dict_size], dtype=tf.float32), name="verb_bias",
                                         trainable=True)

            if self.config.mode == "train":
                print("verb embeddings shape: {}".format(self.get_shape(self.verb_embeddings)), flush=True)

        with tf.variable_scope("negative_sampling"):
            # self.neg_verbs = self._add_negative_samples(self.verb)
            self.neg_verbs, *_ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=tf.cast(tf.expand_dims(self.verb, axis=1), dtype=tf.int64),
                num_true=1,
                num_sampled=self.config.neg_sample,
                unique=True,
                range_max=self.verb_dict_size,
                distortion=0.75,
                num_reserved_ids=1,  # exclude the UNK token
                unigrams=self.verb_count,
                seed=12345))  # [num_neg_samples]

            if self.config.mode == "train":
                print("negative samples shape: {}".format(self.get_shape(self.neg_verbs)), flush=True)

        with tf.variable_scope("embedding_lookup"):
            left_context_emb = tf.nn.embedding_lookup(self.word_embeddings, self.left_context)
            right_context_emb = tf.nn.embedding_lookup(self.word_embeddings, self.right_context)

            verb_emb = tf.nn.embedding_lookup(self.verb_embeddings, self.verb)
            neg_verb_emb = tf.nn.embedding_lookup(self.verb_embeddings, self.neg_verbs)
            verb_emb_bias = tf.nn.embedding_lookup(self.verb_bias, self.verb)
            neg_verb_emb_bias = tf.nn.embedding_lookup(self.verb_bias, self.neg_verbs)

        # left context bidirectional lstm
        with tf.variable_scope("context_left"):
            cell_fw = LSTMCell(num_units=self.config.num_units)
            cell_bw = LSTMCell(num_units=self.config.num_units)

            left_context_features, _ = bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                 cell_bw=cell_bw,
                                                                 inputs=left_context_emb,
                                                                 sequence_length=self.left_seq_len,
                                                                 dtype=tf.float32,
                                                                 scope="bi_lstm")

            left_context_features = tf.concat(left_context_features, axis=-1)

            # self-attention
            left_context_features = self_attention(left_context_features,
                                                   num_units=self.config.num_units,
                                                   init_range=self.config.initializer_range,
                                                   name="self_attention")

            # dense layer project
            left_context_features = tf.layers.dense(left_context_features,
                                                    units=2 * self.config.num_units,
                                                    use_bias=True,
                                                    kernel_initializer=tf.truncated_normal_initializer(
                                                        stddev=self.config.initializer_range),
                                                    bias_initializer=tf.zeros_initializer(),
                                                    activation=self.activation,
                                                    name="dense")

            if self.config.mode == "train":
                print("left context shape: {}".format(self.get_shape(left_context_features)), flush=True)

        # right context bidirectional lstm
        with tf.variable_scope("context_right"):
            cell_fw = LSTMCell(num_units=self.config.num_units)
            cell_bw = LSTMCell(num_units=self.config.num_units)

            right_context_features, _ = bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                  cell_bw=cell_bw,
                                                                  inputs=right_context_emb,
                                                                  sequence_length=self.right_seq_len,
                                                                  dtype=tf.float32,
                                                                  scope="bi_lstm")

            right_context_features = tf.concat(right_context_features, axis=-1)

            # self-attention
            right_context_features = self_attention(right_context_features,
                                                    num_units=self.config.num_units,
                                                    init_range=self.config.initializer_range,
                                                    name="self_attention")

            # dense layer project
            right_context_features = tf.layers.dense(right_context_features,
                                                     units=2 * self.config.num_units,
                                                     use_bias=True,
                                                     kernel_initializer=tf.truncated_normal_initializer(
                                                         stddev=self.config.initializer_range),
                                                     bias_initializer=tf.zeros_initializer(),
                                                     activation=self.activation,
                                                     name="dense")

            if self.config.mode == "train":
                print("right context shape: {}".format(self.get_shape(right_context_features)))

        # concatenation and projection
        with tf.variable_scope("context_fusion"):
            context_features = tf.concat([left_context_features, right_context_features], axis=-1)

            self.context = tf.layers.dense(context_features,
                                           units=self.config.output_units,
                                           use_bias=True,
                                           kernel_initializer=tf.truncated_normal_initializer(
                                               stddev=self.config.initializer_range),
                                           bias_initializer=tf.zeros_initializer(),
                                           activation=self.activation,
                                           name="context_project")

            if self.config.mode == "train":
                print("context representation shape: {}".format(self.get_shape(self.context)), flush=True)

        with tf.variable_scope("verb_representation"):
            self.target_verb = feed_forward_layer(inputs=verb_emb,
                                                  hidden_units=self.config.num_units,
                                                  output_units=self.config.output_units,
                                                  init_range=self.config.initializer_range,
                                                  use_bias=True,
                                                  activation=self.activation,
                                                  reuse=tf.AUTO_REUSE,
                                                  name="feed_forward_layer")

            if self.config.mode == "train":
                print("verb representation shape: {}".format(self.get_shape(self.target_verb)), flush=True)

            negative_verbs = feed_forward_layer(inputs=neg_verb_emb,
                                                hidden_units=self.config.num_units,
                                                output_units=self.config.output_units,
                                                init_range=self.config.initializer_range,
                                                use_bias=True,
                                                activation=self.activation,
                                                reuse=tf.AUTO_REUSE,
                                                name="feed_forward_layer")
            if self.config.mode == "train":
                print("negative verb shape: {}".format(negative_verbs.get_shape().as_list()), flush=True)

        with tf.variable_scope("compute_logits"):
            true_logits = tf.reduce_sum(tf.multiply(x=self.context, y=self.target_verb), axis=1) + verb_emb_bias
            negative_logits = tf.matmul(self.context, negative_verbs, transpose_b=True) + neg_verb_emb_bias

            if self.config.mode == "train":
                print("true logits shape: {}".format(self.get_shape(true_logits)), flush=True)
                print("negative logits shape: {}".format(self.get_shape(negative_logits)), flush=True)

        with tf.variable_scope("compute_loss"):

            true_xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=true_logits,
                                                                labels=tf.ones_like(true_logits))

            negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=negative_logits,
                                                                    labels=tf.zeros_like(negative_logits))

            self.loss = (tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)) / self.config.batch_size

        # create optimizer
        num_warmup_steps = int(self.num_train_steps * self.config.warmup_proportion)
        self.train_op = create_optimizer(self.loss, self.config.lr, self.num_train_steps, num_warmup_steps)

    def train(self, dataset, save_step=20000, print_step=1000, debug_step=5000):
        print("Start training...", flush=True)

        g_step = 0
        for epoch in range(1, self.config.epochs + 1):
            for i, data in enumerate(dataset_iterator(dataset, self.word_dict, self.verb_dict, self.config.batch_size)):

                g_step += 1
                feed_dict = self.get_feed_dict(data, training=True)
                _, losses, neg_verbs = self.sess.run([self.train_op, self.loss, self.neg_verbs],
                                                     feed_dict=feed_dict)
                if (i + 1) % print_step == 0:
                    print("Epoch: {}/{}, Steps: {}, Global Steps: {}, Training Loss: {}".format(
                        epoch, self.config.epochs, i + 1, g_step, "{0:.4f}".format(losses)), flush=True)

                if (i + 1) % debug_step == 0:
                    # for debugging usage
                    test_left_context = "When idle, Dave enjoys"
                    test_right_context = "cake with his sister."
                    test_verb = "eating"
                    candidate_verbs = [x for x in list(self.verb_dict.values())]
                    candidate_verbs.remove(self.verb_dict[UNK])
                    candidate_vecs = self.sess.run(self.target_verb, feed_dict={self.verb: candidate_verbs})
                    self._infer(test_left_context, test_right_context, test_verb, candidate_verbs, candidate_vecs,
                                top_n=20, method="add")

                if g_step % save_step == 0:
                    self.save_session(g_step)

            # save the model
            self.save_session(g_step)

        print("Training process finished...", flush=True)

    def get_candidates(self):
        candidate_verbs = [x for x in list(self.verb_dict.values())]
        candidate_verbs.remove(self.verb_dict[UNK])
        candidate_vecs = self.sess.run(self.target_verb, feed_dict={self.verb: candidate_verbs})
        return candidate_verbs, candidate_vecs

    def _infer(self, left_context, right_context, verb, candidate_verbs, candidate_vecs, top_n=10, method="multiply"):
        # pre-process inputs
        processed_tokens = convert_single(left_context, right_context, verb, word_dict=self.word_dict,
                                          verb_dict=self.verb_dict)

        # compute context, target verb and candidates representations in the embedding hyperspace
        context_vec, verb_vec = self.sess.run([self.context, self.target_verb],
                                              feed_dict={
                                                  self.left_context: processed_tokens["l_context"],
                                                  self.left_seq_len: processed_tokens["l_seq_len"],
                                                  self.right_context: processed_tokens["r_context"],
                                                  self.right_seq_len: processed_tokens["r_seq_len"],
                                                  self.verb: processed_tokens["verb"]})

        cv = [float("{:.3f}".format(x)) for x in np.reshape(context_vec, (context_vec.shape[1], )).tolist()]
        vv = [float("{:.3f}".format(x)) for x in np.reshape(verb_vec, (verb_vec.shape[1], )).tolist()]

        similarities = compute_similarity(candidate_vecs, context_vec, verb_vec, method=method)

        candidate_dict = dict()
        for i in range(similarities.shape[0]):
            candidate_dict[candidate_verbs[i]] = similarities[i]

        top_candidates = sorted(candidate_dict.items(), key=lambda kv: kv[1], reverse=True)[0:top_n + 1]
        top_candidates = [self.rev_verb_dict[x] for x, _ in top_candidates]

        print(cv, flush=True)
        print(vv, flush=True)
        print(top_candidates[1:], flush=True)

    def inference(self, left_contexts, right_contexts, verbs, candidate_verbs, candidate_vecs, top_n=10, method="add"):
        # pre-process inputs
        processed_tokens = convert_batch(left_contexts, right_contexts, verbs, self.word_dict, self.verb_dict)

        # compute context, target verb and candidates representations in the embedding hyperspace
        context_vec, verb_vec = self.sess.run([self.context, self.target_verb],
                                              feed_dict={
                                                  self.left_context: processed_tokens["l_context"],
                                                  self.left_seq_len: processed_tokens["l_seq_len"],
                                                  self.right_context: processed_tokens["r_context"],
                                                  self.right_seq_len: processed_tokens["r_seq_len"],
                                                  self.verb: processed_tokens["verb"]})

        # compute cosine similarity (batch_size x num_candidates)
        similarities = compute_similarity_batch(candidate_vecs, context_vec, verb_vec, method=method)

        candidates = []
        for similarity in similarities:
            candidate_dict = dict()
            for i in range(similarity.shape[0]):
                candidate_dict[candidate_verbs[i]] = similarity[i]
            top_candidates = sorted(candidate_dict.items(), key=lambda kv: kv[1], reverse=True)[0:top_n + 1]
            top_candidates = [self.rev_verb_dict[x] for x, _ in top_candidates][1:]
            candidates.extend(top_candidates)
        return candidates


def compute_similarity(candidate_vecs, context_vec, verb_vec, method="add"):
    verb_candidate_similarity = cosine_similarity(verb_vec, candidate_vecs)
    context_candidate_similarity = cosine_similarity(context_vec, candidate_vecs)

    if method == "multiply":
        similarities = verb_candidate_similarity * context_candidate_similarity
        similarities = np.reshape(similarities, newshape=(similarities.shape[1],))
    elif method == "add":
        similarities = verb_candidate_similarity + context_candidate_similarity
        similarities = np.reshape(similarities, newshape=(similarities.shape[1],))
    elif method == "both":
        sim_mul = verb_candidate_similarity * context_candidate_similarity
        sim_mul = np.reshape(sim_mul, newshape=(sim_mul.shape[1],))
        sim_mul = (sim_mul - np.min(sim_mul)) / (np.max(sim_mul) - np.min(sim_mul))  # normalize
        sim_add = verb_candidate_similarity + context_candidate_similarity
        sim_add = np.reshape(sim_add, newshape=(sim_add.shape[1],))
        sim_add = (sim_add - np.min(sim_add)) / (np.max(sim_add) - np.min(sim_add))  # normalize

        similarities = sim_mul + sim_add
    else:
        raise ValueError("Unsupported similarity method, only [multiply | add | both] are allowed...")

    return similarities


def compute_similarity_batch(candidate_vecs, context_vec, verb_vec, method="add"):
    verb_candidate_similarity = cosine_similarity(verb_vec, candidate_vecs)
    context_candidate_similarity = cosine_similarity(context_vec, candidate_vecs)

    if method == "multiply":
        similarities = verb_candidate_similarity * context_candidate_similarity
    elif method == "add":
        similarities = verb_candidate_similarity + context_candidate_similarity
    elif method == "both":
        sim_mul = verb_candidate_similarity * context_candidate_similarity
        # TODO: normalize
        sim_add = verb_candidate_similarity + context_candidate_similarity
        # TODO: normalize
        similarities = sim_mul + sim_add
    else:
        raise ValueError("Unsupported similarity method, only [multiply | add | both] are allowed...")

    return similarities
