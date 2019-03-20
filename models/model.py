import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models.ops_nns import feed_forward_nets, neural_tensor_net, context_representation, naive_fusion, \
    embedding_lookup, compute_loss, compute_top_candidates
from utils.data_utils import dataset_iterator, load_pickle, write_pickle, convert_single
from utils.data_prepro import UNK


class ConceptualPrimitives:
    def __init__(self, cfg, verb_count, word_dict, verb_dict):
        self.cfg = cfg
        self.verb_count = verb_count
        self.word_dict = word_dict
        self.word_dict_size = len(self.word_dict)
        self.verb_dict = verb_dict
        self.verb_dict_size = len(self.verb_dict)
        self.rev_verb_dict = dict([(idx, verb) for verb, idx in self.verb_dict.items()])

        self._add_placeholders()
        self._build_model()

        if self.cfg.mode == "train":
            print("total params: {}".format(self.count_params()), flush=True)
        self._init_session()

    def _init_session(self):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self.saver = tf.train.Saver(max_to_keep=self.cfg.max_to_keep)

        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)

        else:
            ckpt = tf.train.get_checkpoint_state(self.cfg.ckpt)  # get checkpoint state

        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored model from %s, done..." % ckpt.model_checkpoint_path)

        else:
            raise ValueError("no pretrained parameters in %s directory, please check or start a new session to train" %
                             self.cfg.ckpt)

    def save_session(self, global_step):
        self.saver.save(sess=self.sess,
                        save_path=self.cfg.ckpt + self.cfg.model_name,
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

    def _add_negative_samples(self, inputs):
        # cf. https://github.com/carpedm20/practice-tensorflow/blob/master/embedding/word2vec.py
        neg_verbs, *_ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.cast(tf.expand_dims(inputs, axis=1), dtype=tf.int64),
            num_true=1,
            num_sampled=self.cfg.batch_size * self.cfg.neg_sample,
            unique=False,
            range_max=self.verb_dict_size,
            distortion=self.cfg.distortion,
            num_reserved_ids=1,  # exclude the UNK token
            unigrams=self.verb_count,
            seed=12345
        ))
        return tf.reshape(neg_verbs, shape=[self.cfg.batch_size, self.cfg.neg_sample])

    def _build_model(self):
        with tf.device("/gpu:{}".format(self.cfg.gpu_idx[0])):
            # word embedding lookup
            left_context_emb, right_context_emb, self.word_embeddings = embedding_lookup(inputs1=self.left_context,
                                                                                         inputs2=self.right_context,
                                                                                         word_dim=self.cfg.word_dim,
                                                                                         vectors=self.cfg.word_vectors,
                                                                                         tune_emb=self.cfg.tune_emb,
                                                                                         use_pad=True,
                                                                                         name="word_embedding_lookup")

            if self.cfg.mode == "train":
                print("word embeddings shape: {}".format(self.get_shape(self.word_embeddings)), flush=True)
                print("left context embedding shape: {}".format(self.get_shape(left_context_emb)), flush=True)
                print("right context embedding shape: {}".format(self.get_shape(right_context_emb)), flush=True)

            # verb embedding lookup
            with tf.variable_scope("negative_sampling"):
                self.neg_verbs = self._add_negative_samples(self.verb)

            verb_emb, neg_verb_emb, self.verb_embeddings = embedding_lookup(inputs1=self.verb,
                                                                            inputs2=self.neg_verbs,
                                                                            word_dim=self.cfg.word_dim,
                                                                            vectors=self.cfg.verb_vectors,
                                                                            tune_emb=self.cfg.tune_emb,
                                                                            use_pad=False,
                                                                            name="verb_embedding_lookup")

            if self.cfg.mode == "train":
                print("verb embeddings shape: {}".format(self.get_shape(self.verb_embeddings)), flush=True)
                print("verb inputs shape: {}".format(self.get_shape(verb_emb)), flush=True)
                print("negative verb inputs shape: {}".format(self.get_shape(neg_verb_emb)), flush=True)

            # compute left context
            left_context_features = context_representation(inputs=left_context_emb,
                                                           seq_len=self.left_seq_len,
                                                           num_units=self.cfg.num_units,
                                                           activation=tf.nn.tanh,
                                                           use_bias=False,
                                                           reuse=False,
                                                           name="context_rep_left")

            if self.cfg.mode == "train":
                print("left context shape: {}".format(self.get_shape(left_context_features)), flush=True)

            # compute right context
            right_context_features = context_representation(inputs=right_context_emb,
                                                            seq_len=self.right_seq_len,
                                                            num_units=self.cfg.num_units,
                                                            activation=tf.nn.tanh,
                                                            use_bias=False,
                                                            reuse=False,
                                                            name="context_rep_right")
            if self.cfg.mode == "train":
                print("right context shape: {}".format(self.get_shape(right_context_features)))

        with tf.device("/gpu:{}".format(self.cfg.gpu_idx[1] if self.cfg.use_ntn else self.cfg.gpu_idx[0])):
            if self.cfg.use_ntn:
                # neural tensor network
                self.context = neural_tensor_net(inputs1=left_context_features,
                                                 inputs2=right_context_features,
                                                 hidden_units=2 * self.cfg.num_units,
                                                 output_units=self.cfg.k,
                                                 activation=tf.nn.tanh,
                                                 reuse=False,
                                                 name="neural_tensor_network")

            else:
                # concatenation and projection
                self.context = naive_fusion(inputs1=left_context_features,
                                            inputs2=right_context_features,
                                            output_units=self.cfg.k,
                                            use_bias=True,
                                            bias_init=0.0,
                                            activation=tf.nn.tanh,
                                            reuse=False,
                                            name="naive_fusion")

            if self.cfg.mode == "train":
                print("context representation shape: {}".format(self.get_shape(self.context)), flush=True)

            with tf.variable_scope("verb_representation"):
                self.target_verb = feed_forward_nets(inputs=verb_emb,
                                                     hidden_units=self.cfg.num_units,
                                                     output_units=self.cfg.k,
                                                     use_bias=True,
                                                     bias_init=0.0,
                                                     activation=tf.nn.tanh,
                                                     reuse=tf.AUTO_REUSE,
                                                     name="feed_forward_layer")

                if self.cfg.mode == "train":
                    print("verb representation shape: {}".format(self.get_shape(self.target_verb)), flush=True)

                negative_verbs = feed_forward_nets(inputs=neg_verb_emb,
                                                   hidden_units=self.cfg.num_units,
                                                   output_units=self.cfg.k,
                                                   use_bias=True,
                                                   bias_init=0.0,
                                                   activation=tf.nn.tanh,
                                                   reuse=tf.AUTO_REUSE,
                                                   name="feed_forward_layer")
                if self.cfg.mode == "train":
                    print("negative verb shape: {}".format(negative_verbs.get_shape().as_list()), flush=True)

            # compute loss
            true_logits, negative_logits, self.loss = compute_loss(verbs=self.target_verb,
                                                                   neg_verbs=negative_verbs,
                                                                   context=self.context,
                                                                   batch_size=self.cfg.batch_size,
                                                                   name="compute_loss")
            if self.cfg.mode == "train":
                print("true logits shape: {}".format(self.get_shape(true_logits)), flush=True)
                print("negative logits shape: {}".format(self.get_shape(negative_logits)), flush=True)

            # build optimizer
            global_step = tf.Variable(0, trainable=False, name='global_step')
            learning_rate = tf.train.exponential_decay(learning_rate=self.cfg.lr,
                                                       global_step=global_step,
                                                       decay_steps=self.cfg.decay_step,
                                                       decay_rate=self.cfg.decay_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=global_step)
    
    def get_verb_embs(self, emb_type="init", save_path=None):
        if save_path is not None and os.path.exists(save_path):
            result = load_pickle(filename=save_path)
            return result["vocab"], result["vectors"]

        verb_dict = self.verb_dict.copy()
        verb_dict.pop(UNK)  # remove UNK

        if emb_type == "init":
            verb_embs = self.sess.run(self.verb_embeddings)[1:]  # remove UNK
            verb_vocab, verb_vectors = list(), list()

            for verb, idx in tqdm(verb_dict.items(), total=len(verb_dict), desc="extract verb embeddings"):
                verb_vocab.append(verb)
                verb_vectors.append(verb_embs[idx])

            result = {"vocab": verb_vocab, "vectors": np.asarray(verb_vectors)}

        elif emb_type == "target":
            verb_vocab, verb_vectors = list(), list()

            for verb, idx in tqdm(verb_dict.items(), total=len(verb_dict), desc="extract verb representations"):
                verb_vector = self.sess.run(self.target_verb, feed_dict={self.verb: [idx]})
                verb_vocab.append(verb)
                verb_vectors.append(np.reshape(verb_vector, newshape=(self.cfg.k, )))

            result = {"vocab": verb_vocab, "vectors": np.asarray(verb_vectors)}

        else:
            raise ValueError("Unknown emb type...")

        if save_path is not None:
            write_pickle(result, filename=save_path)

        return result["vocab"], result["vectors"]

    def train(self, dataset, save_step=10000, print_step=1000):
        print("Start training...", flush=True)

        global_step = 0
        mean_losses = []
        for epoch in range(1, self.cfg.epochs + 1):
            for i, data in enumerate(dataset_iterator(dataset, self.word_dict, self.verb_dict, self.cfg.batch_size)):

                global_step += 1
                feed_dict = self.get_feed_dict(data, training=True)
                _, losses, neg_verbs = self.sess.run([self.train_op, self.loss, self.neg_verbs], feed_dict=feed_dict)
                mean_losses.append(losses)

                if (i + 1) % print_step == 0:
                    train_loss = "{0:.4f}".format(losses)
                    print("Epoch: {}/{}, Cur Steps: {}, Global Steps: {}, Cur Training Loss: {}, Mean Loss: {}".format(
                        epoch, self.cfg.epochs, i + 1, global_step, train_loss, np.mean(mean_losses)), flush=True)
                    # for debugging usage
                    test_sentence = "When idle, Dave enjoys eating cake with his sister."
                    test_verb = "eating"
                    candidates = self.inference(test_sentence, test_verb, top_n=20, method="add", show_vec=True)
                    print(candidates)

                if global_step % save_step == 0:
                    self.save_session(global_step)

            # save the model
            self.save_session(global_step)

        print("Training process finished...", flush=True)

    def inference(self, sentence, verb, top_n=10, method="multiply", show_vec=True):
        # pre-process inputs
        processed_tokens = convert_single(sentence, verb, word_dict=self.word_dict, verb_dict=self.verb_dict)

        # compute context, target verb and candidates representations in the embedding hyperspace
        context_vec, verb_vec = self.sess.run([self.context, self.target_verb],
                                              feed_dict={
                                                  self.left_context: processed_tokens["l_context"],
                                                  self.left_seq_len: processed_tokens["l_seq_len"],
                                                  self.right_context: processed_tokens["r_context"],
                                                  self.right_seq_len: processed_tokens["r_seq_len"],
                                                  self.verb: processed_tokens["verb"]
                                              })

        if show_vec is True:
            cv = [float("{:.3f}".format(x)) for x in np.reshape(context_vec, newshape=(context_vec.shape[1],)).tolist()]
            vv = [float("{:.3f}".format(x)) for x in np.reshape(verb_vec, newshape=(verb_vec.shape[1],)).tolist()]
            print("context vec: {}\nverb vec: {}".format(str(cv), str(vv)), flush=True)

        candidate_verbs = processed_tokens["candidate_verbs"]
        candidate_vecs = self.sess.run(self.target_verb, feed_dict={self.verb: candidate_verbs})

        top_candidates = compute_top_candidates(candidate_verbs=candidate_verbs,
                                                candidate_vecs=candidate_vecs,
                                                context_vec=context_vec,
                                                verb_vec=verb_vec,
                                                rev_dict=self.rev_verb_dict,
                                                method=method,
                                                top_n=top_n)
        return top_candidates
