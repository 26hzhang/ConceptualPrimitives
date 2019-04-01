import os
import pickle
import codecs
from nltk.tokenize import word_tokenize
from utils.data_prepro import UNK, separator


def boolean_string(bool_str):
    bool_str = bool_str.lower()

    if bool_str not in {"false", "true"}:
        raise ValueError("Not a valid boolean string!!!")

    return bool_str == "true"


def save_vectors_to_file(vectors, save_name):
    with codecs.open(save_name, mode="w", encoding="utf-8") as f:
        for i in range(vectors.shape[0]):
            vector = " ".join([str(x) for x in vectors[i, :].tolist()])
            f.write("{}\n\n".format(vector))


def load_vector_pickle(filename):
    with open(filename, mode="rb") as handle:
        data = pickle.load(handle)
    vocab, vectors = data["vocab"], data["vectors"]
    return vocab, vectors


def load_pickle(filename):
    with open(filename, mode="rb") as handle:
        data = pickle.load(handle)
    return data


def write_pickle(data, filename):
    with open(filename, mode="wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_csv(cluster_dict, save_path):
    with codecs.open(save_path, mode="w", encoding="utf-8") as f:
        for key, value in cluster_dict.items():
            f.write("{},{}\n".format(key, ",".join(value)))


def read_to_dict(filename):
    vocab = dict()

    with codecs.open(filename, mode="r", encoding="utf-8") as f:
        for idx, word in enumerate(f):
            word = word.lstrip().rstrip()
            vocab[word] = idx

    return vocab


def read_verb_count(filename):
    count_list = []

    with codecs.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            count = int(line[1])
            count_list.append(count)

    return count_list


def pad_sequence(sequence, pad_tok=0, max_length=None):
    """Pad batched dataset with shape = (batch_size, seq_length(various))
    :param sequence: input sequence
    :param pad_tok: padding token, default is 0
    :param max_length: max length of padded sequence, default is None
    :return: padded sequence
    """
    if max_length is None:
        max_length = max([len(seq) for seq in sequence])

    sequence_padded, seq_length = [], []
    for seq in sequence:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        seq_length.append(min(len(seq), max_length))

    return sequence_padded, seq_length


def build_batch_dataset(left_context, verbs, right_context):
    left_context, left_seq_len = pad_sequence(left_context)
    right_context, right_seq_len = pad_sequence(right_context)
    batch_size = len(verbs)

    return {"l_context": left_context, "l_seq_len": left_seq_len, "r_context": right_context,
            "r_seq_len": right_seq_len, "verb": verbs, "batch_size": batch_size}


def dataset_iterator(dataset_file, word_dict, verb_dict, batch_size):
    if dataset_file is None or not os.path.exists(dataset_file):
        raise IOError("Unable to find the dataset file: %s" % dataset_file)

    with codecs.open(dataset_file, mode="r", encoding="utf-8", errors="ignore") as f_dataset:
        left_contexts, verbs, right_contexts = [], [], []

        for line in f_dataset:
            # split data
            l_context, verb, r_context = line.strip().split(separator)

            # convert to indices
            l_context = [word_dict[word] if word in word_dict else word_dict[UNK] for word in l_context.split(" ")]
            verb = verb_dict[verb] if verb in verb_dict else verb_dict[UNK]
            r_context = [word_dict[word] if word in word_dict else word_dict[UNK] for word in r_context.split(" ")]

            # add to list
            left_contexts.append(l_context)
            verbs.append(verb)
            right_contexts.append(r_context)

            # yield batched dataset
            if len(left_contexts) == batch_size:
                yield build_batch_dataset(left_contexts, verbs, right_contexts)
                left_contexts, verbs, right_contexts = [], [], []


def convert_single(left_sent, right_sent, verb, word_dict, verb_dict):
    # pre-process inputs
    verb = verb.lower()
    left_words = word_tokenize(left_sent.lower(), language="english")
    right_words = word_tokenize(right_sent.lower(), language="english")

    # build feed data
    l_context = [word_dict[w] if w in word_dict else word_dict[UNK] for w in left_words]
    l_seq_len = len(l_context)
    r_context = [word_dict[w] if w in word_dict else word_dict[UNK] for w in right_words]
    r_seq_len = len(r_context)
    verb = verb_dict[verb] if verb in verb_dict else verb_dict[UNK]

    # deal with substitution of target verb
    candidate_verbs = [x for x in list(verb_dict.values())]
    candidate_verbs.remove(verb_dict[UNK])
    candidate_verbs.remove(verb)

    return {"l_context": [l_context], "l_seq_len": [l_seq_len], "r_context": [r_context],
            "r_seq_len": [r_seq_len], "verb": [verb], "candidate_verbs": candidate_verbs}
