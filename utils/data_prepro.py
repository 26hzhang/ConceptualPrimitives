import os
import codecs
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.corpus import stopwords

PAD = "<PAD>"  # pad token
UNK = "<UNK>"  # unknown token
separator = "\t"  # split left context, target verb and right context

# the english stopwords list used for remove unwanted/wrong target verbs
stopwords_list = set(list(stopwords.words("english")) + ["'s", "'ve", "'re", "'m", "'d", "don\\'t", "don''t", "did'nt",
                                                         "can''t", "24-hours", "you.", "e.g.", "a.", "said.", "be.",
                                                         "ie.", "i.e", "do.", "eg.", "i.", "....", "c.", "viz.", "b.",
                                                         "e.g", "f.", "ur", "il", "help.", "java.lang.string", "have.",
                                                         "go.", "see.", ".....", "know.", ".", "luv", "it.", "ia",
                                                         "use.", "think.", "need.", "is.", "are.", "happen.", "cf.",
                                                         "esp.", "me.", "apply.", "want.", "n.", "approx.", "s.",
                                                         "expect.", "change.", "c/o", "can.", "and/or", "iron/ironing",
                                                         "lounge/dining", "living/dining", "pf", "n/a", "to/from", "hi",
                                                         "see/hear"])

# cf. https://courses.washington.edu/hypertxt/csar-v02/penntable.html
# cf. http://wacky.sslmit.unibo.it/lib/exe/fetch.php?media=tagsets:ukwac_tagset.txt
verb_pos = ["VB", "VBZ", "VBP", "VBD", "VBN", "VBG", "VV", "VVN", "VVG", "VVD", "VVZ", "VVP", "VH", "VHP", "VHZ", "VHD",
            "VHG", "VHN"]

# approximately vocabulary size for different pretrained glove vectors
glove_size = {"2B": int(1.2e6), "6B": int(4e5), "42B": int(1.9e6), "840B": int(2.2e6)}


def load_glove_vocabulary(glove_path, dim):
    vocab = list()

    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:

        for line in tqdm(f, total=glove_size[glove_path.split(".")[-3]], desc="Load GloVe vocabulary"):
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
        for line in tqdm(f, total=glove_size[glove_path.split(".")[-3]], desc="Load GloVe embeddings"):
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
                    if len(words) > 80:  # if sentence length is greater than 80, then ignore this sentence
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


def build_dataset(ukwac_path, glove_path, save_path, word_threshold=90, word_lowercase=True, word_dim=300):
    word_counter, verb_counter = Counter(), Counter()

    dataset_save_path = os.path.join(save_path, "dataset.txt")
    temp_dataset_recorder = list()

    with codecs.open(dataset_save_path, mode="w", encoding="utf-8") as f:
        for words, pos_tags, lemma_words in ukwac_corpus_iterator(ukwac_path, word_lowercase):
            assert len(words) == len(pos_tags), "the size of words and corresponding pos tags must be equal"

            for index, (word, pos_tag, lemma_word) in enumerate(zip(words, pos_tags, lemma_words)):
                word_counter[word] += 1  # count word frequency

                if index != 0 and index != len(words) - 1 and pos_tag in verb_pos and word not in stopwords_list:
                    verb_counter[word] += 1  # count valid verb frequency
                    l_context = " ".join(words[:index])
                    r_context = " ".join(words[index + 1:])
                    result = separator.join([l_context, word, r_context])
                    temp_dataset_recorder.append(result)

            if len(temp_dataset_recorder) > int(1e6):
                random.shuffle(temp_dataset_recorder)

                while len(temp_dataset_recorder) > 0:
                    result = temp_dataset_recorder.pop()
                    f.write(result + "\n")

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
