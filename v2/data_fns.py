import os
import pickle
import codecs
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

PAD = "<PAD>"  # pad token
UNK = "<UNK>"  # unknown token
SEPARATOR = "\t"  # split left context, target verb and right context

# the english stopwords list used for remove unwanted target verbs
stopwords_list = set(list(stopwords.words("english")) + ["'s", "'ve", "'re", "'m", "'d", "don\\'t", "don''t", "did'nt",
                                                         "can''t", "24-hours", "you.", "e.g.", "a.", "said.", "be.",
                                                         "ie.", "i.e", "do.", "eg.", "i.", "....", "c.", "viz.", "b.",
                                                         "e.g", "f.", "ur", "il", "help.", "java.lang.string", "have.",
                                                         "go.", "see.", ".....", "know.", ".", "luv", "it.", "ia",
                                                         "use.", "think.", "need.", "is.", "are.", "happen.", "cf.",
                                                         "esp.", "me.", "apply.", "want.", "n.", "approx.", "s.",
                                                         "expect.", "change.", "c/o", "can.", "and/or", "iron/ironing",
                                                         "lounge/dining", "living/dining", "pf", "n/a", "to/from", "hi",
                                                         "see/hear", "pkt", "rel", "ans", "rss", "can?t", "don?t",
                                                         "didn?t", "didnt", "dont", "doesnt", "isnt", "wasnt",
                                                         "shouldnt", "hadnt", "hasnt", "havent", "wont", "shanghai",
                                                         "beene", "mak", "nave", "co", "http", "web", "youre", ":p",
                                                         "ws", "sic", "eh", "whan", "def", ":d", "https", "can",
                                                         "cannot", "could", "should"])

# cf. https://courses.washington.edu/hypertxt/csar-v02/penntable.html
# cf. http://wacky.sslmit.unibo.it/lib/exe/fetch.php?media=tagsets:ukwac_tagset.txt
'''
VB	be, base (be)
VBD	be, past tense (went)
VBG	be, -ing (being)
VBN	be, past participle (been)
VBP	be, plural (are)
VBZ	be, -s (is)

VH	have, base (have)
VHD	have, past tense (had)
VHG	have, -ing (having)
VHN	have, past participle (had)
VHP	have, plural (have)

VHZ	verb, -s (believes)
VV	verb, base (believe)
VVD	verb, past tense (believed)
VVG	verb, -ing (believing)
VVN	verb, past participle (believed)
VVP	verb, plural (believe)
VBZ	verb, -s (believes)
'''

# all verb pos
verb_pos = ["VB", "VBZ", "VBP", "VBD", "VBN", "VBG", "VV", "VVN", "VVG", "VVD", "VVZ", "VVP", "VH", "VHP", "VHZ", "VHD",
            "VHG", "VHN"]

# approximately vocabulary size for different pretrained glove vectors
glove_size = {"2B": int(1.2e6), "6B": int(4e5), "42B": int(1.9e6), "840B": int(2.2e6)}


def boolean_string(bool_str):
    bool_str = bool_str.lower()

    if bool_str not in {"false", "true"}:
        raise ValueError("Not a valid boolean string!!!")

    return bool_str == "true"


def compute_num_file_lines(filename):
    count = 0
    with open(filename, mode="r", encoding="utf-8") as f:
        for line in f:
            if len(line) == 0:
                continue
            count += 1
    return count


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


def load_glove_vocabulary(glove_path, dim):
    vocab = list()

    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=glove_size["840B"], desc="Load GloVe vocabulary"):
            line = line.lstrip().rstrip().split(" ")

            if len(line) == 2 or len(line) != dim + 1:
                continue

            word = line[0]
            vocab.append(word)

    return set(vocab)


def load_glove_vectors(glove_path, word_dict, verb_dict, dim):
    word_vectors = np.zeros(shape=[len(word_dict), dim], dtype=np.float32)
    verb_vectors = np.zeros(shape=[len(verb_dict), dim], dtype=np.float32)

    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=glove_size["840B"], desc="Load GloVe embeddings"):
            line = line.lstrip().rstrip().split(" ")

            if len(line) == 2 or len(line) != dim + 1:
                continue

            word = line[0]

            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_index = word_dict[word]
                word_vectors[word_index] = np.asarray(vector)

            if word in verb_dict:
                vector = [float(x) for x in line[1:]]
                verb_index = verb_dict[word]
                verb_vectors[verb_index] = np.asarray(vector)

    return word_vectors, verb_vectors


def ukwac_corpus_iterator(ukwac_path, lowercase=True):
    if not os.path.exists(ukwac_path):
        raise IOError("Unable to find the corpus directory: %s" % ukwac_path)

    ukwac_file = os.path.join(ukwac_path, "UKWAC-{}.xml")
    words, pos_tags, lemma_words = [], [], []

    for index in range(1, 26):
        total_lines = int(3.336e7) if index == 25 else int(1e8)

        with codecs.open(ukwac_file.format(index), mode="r", encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f, total=total_lines, desc="Read UKWAC-{}.xml".format(index)):
                line = line.lstrip().rstrip()
                if line.startswith("<text id") or line.startswith("<s>") or line.startswith("</text>"):  # start marks
                    continue

                if line.startswith("</s>"):  # finish to read one sentence
                    # if len(words) < 5:  # if sentence length is smaller than 5, then ignore this sentence
                    if len(words) < 6:  # if sentence length is smaller than 6, then ignore this sentence
                        words, pos_tags, lemma_words = [], [], []
                        continue
                    # if len(words) > 60:  # if sentence length is greater than 60, then ignore this sentence
                    if len(words) > 25:  # if sentence length is greater than 25, then ignore this sentence
                        words, pos_tags, lemma_words = [], [], []
                        continue
                    yield words, pos_tags, lemma_words
                    words, pos_tags, lemma_words = [], [], []
                    continue

                if len(line.split("\t")) != 3:
                    continue

                word, pos_tag, lemma_word = line.split("\t")
                words.append(word.strip().lower() if lowercase else word.strip())
                pos_tags.append(pos_tag.strip())
                lemma_words.append(lemma_word.strip().lower() if lowercase else lemma_word.strip())

    if len(words) > 0:
        yield words, pos_tags, lemma_words


def build_dataset(ukwac_path, glove_path, save_path, word_threshold=90, word_lowercase=True, use_lemma=False,
                  word_dim=300):
    word_counter, verb_counter = Counter(), Counter()

    dataset_save_path = os.path.join(save_path, "dataset.txt")
    temp_dataset_recorder = list()

    with codecs.open(dataset_save_path, mode="w", encoding="utf-8") as f:
        for raw_words, pos_tags, lemma_words in ukwac_corpus_iterator(ukwac_path, word_lowercase):
            assert len(raw_words) == len(lemma_words) == len(pos_tags), "the size of each column must be equal"

            words = lemma_words if use_lemma else raw_words

            for index, (word, pos_tag) in enumerate(zip(words, pos_tags)):
                word_counter[word] += 1  # count word frequency

                if index != 0 and index != len(words) - 1 and pos_tag in verb_pos and word not in stopwords_list:
                    verb_counter[word] += 1  # count valid verb frequency
                    l_context = " ".join(words[:index])
                    r_context = " ".join(words[index + 1:])
                    result = SEPARATOR.join([l_context, word, r_context])
                    temp_dataset_recorder.append(result)

            # save processed sentences
            if len(temp_dataset_recorder) > int(1e6):
                random.shuffle(temp_dataset_recorder)

                while len(temp_dataset_recorder) > 0:
                    result = temp_dataset_recorder.pop()
                    f.write(result + "\n")
                    f.flush()

                temp_dataset_recorder = list()

    # load pretrained glove vocabulary
    glove_vocab = load_glove_vocabulary(glove_path, word_dim)

    # build word vocab and dict
    word_vocab = [word for word, count in word_counter.most_common() if count >= word_threshold and word in glove_vocab]
    word_dict = dict([(word, index) for index, word in enumerate(word_vocab)])
    word_vocab = [PAD, UNK] + word_vocab

    # build verb vocab and dict
    verb_count = list()
    unk_count = 0
    for verb, count in verb_counter.most_common():
        if count >= word_threshold and verb in glove_vocab:
            verb_count.append((verb, count))
        else:
            unk_count += count
    verb_dict = dict([(verb[0], index) for index, verb in enumerate(verb_count)])
    verb_count = [(UNK, unk_count)] + verb_count
    verb_vocab = [v[0] for v in verb_count]

    verb_count = verb_count[1:]  # remove UNK count from count list

    # load pretrained glove vectors
    word_vectors, verb_vectors = load_glove_vectors(glove_path, word_dict, verb_dict, word_dim)
    np.savez_compressed(os.path.join(save_path, "word_vectors.npz"), embeddings=word_vectors)
    np.savez_compressed(os.path.join(save_path, "verb_vectors.npz"), embeddings=verb_vectors)

    with codecs.open(os.path.join(save_path, "word_vocab.txt"), mode="w", encoding="utf-8") as f:
        f.write("\n".join(word_vocab))

    with codecs.open(os.path.join(save_path, "verb_vocab.txt"), mode="w", encoding="utf-8") as f:
        f.write("\n".join(verb_vocab))

    with codecs.open(os.path.join(save_path, "verb_count.txt"), mode="w", encoding="utf-8") as f:
        f.write("\n".join(["{}\t{}".format(verb, count) for verb, count in verb_count]))


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


def convert_single(left_context, right_context, verb, word_dict, verb_dict):
    # pre-process inputs
    verb = verb.lower()
    left_context = word_tokenize(left_context.lower(), language="english")
    right_context = word_tokenize(right_context.lower(), language="english")

    # build feed data
    l_context = [word_dict[w] if w in word_dict else word_dict[UNK] for w in left_context]
    l_seq_len = len(l_context)
    r_context = [word_dict[w] if w in word_dict else word_dict[UNK] for w in right_context]
    r_seq_len = len(r_context)
    verb = verb_dict[verb] if verb in verb_dict else verb_dict[UNK]

    return {"l_context": [l_context], "l_seq_len": [l_seq_len], "r_context": [r_context],
            "r_seq_len": [r_seq_len], "verb": [verb]}


def convert_batch(left_contexts, right_contexts, verbs, word_dict, verb_dict):
    left_contexts = [word_tokenize(context.lower(), language="english") for context in left_contexts]
    verbs = [verb.lower() for verb in verbs]
    right_contexts = [word_tokenize(context.lower(), language="english") for context in right_contexts]

    l_contexts, r_contexts, verbs_ = [], [], []
    for left_context, right_context, verb in zip(left_contexts, right_contexts, verbs):
        l_context = [word_dict[w] if w in word_dict else word_dict[UNK] for w in left_context]
        r_context = [word_dict[w] if w in word_dict else word_dict[UNK] for w in right_context]
        verb = verb_dict[verb] if verb in verb_dict else verb_dict[UNK]
        l_contexts.append(l_context)
        r_contexts.append(r_context)
        verbs_.append(verb)

    l_contexts, l_seq_len = pad_sequence(l_contexts)
    r_contexts, r_seq_len = pad_sequence(r_contexts)
    batch_size = len(verbs)

    return {"l_context": l_contexts, "l_seq_len": l_seq_len, "r_context": r_contexts, "r_seq_len": r_seq_len,
            "verb": verbs_, "batch_size": batch_size}


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
            l_context, verb, r_context = line.strip().split(SEPARATOR)

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


def extract_sentences_for_verb(verb, dataset_path):
    directory = "data/sentences/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = os.path.join(directory, verb + ".txt")
    if os.path.exists(save_path):
        return save_path
    with open(save_path, mode="w", encoding="utf-8") as f_out:
        with open(dataset_path, mode="r", encoding="utf-8") as f:
            for line in tqdm(f, total=166000373, desc="extract sentences of {}".format(verb)):
                line = line.lstrip().rstrip()
                # TODO -- add sentence length threshold
                # normally 5~20 words
                _, cur_verb, _ = line.split(SEPARATOR)
                cur_verb = cur_verb.lstrip().rstrip()
                if cur_verb == verb:
                    f_out.write(line + "\n")
                    f_out.flush()
    return save_path
