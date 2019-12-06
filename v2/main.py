import os
import spacy
from tqdm import tqdm
from collections import Counter
from argparse import ArgumentParser
from model import ConceptualPrimitives
from data_fns import boolean_string, read_to_dict, read_verb_count, build_dataset
from data_fns import extract_sentences_for_verb, SEPARATOR, compute_num_file_lines

home = os.path.expanduser("~")

# set network config
parser = ArgumentParser()
parser.add_argument("--gpu_idx", type=int, default=0, help="GPUs used for training")
parser.add_argument("--mode", type=str, default="train", help="mode [train | cluster | infer], default, train")
parser.add_argument("--random_seed", type=int, default=12345, help="random seed")
parser.add_argument("--resume_training", type=boolean_string, default=False, help="resume previous trained parameters")
parser.add_argument("--use_neg_sample", type=boolean_string, default=True, help="")
parser.add_argument("--neg_sample", type=int, default=10, help="number of negative samples")
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--num_units", type=int, default=300, help="number of units for rnn cell and hidden layer of ffn")
parser.add_argument("--output_units", type=int, default=300, help="number of units for output part")
parser.add_argument("--tune_emb", type=boolean_string, default=True, help="tune pretrained embeddings while training")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="warmup proportion")
parser.add_argument("--batch_size", type=int, default=400, help="batch size")
parser.add_argument("--initializer_range", type=float, default=0.02, help="weight initializer range")
parser.add_argument("--epochs", type=int, default=20, help="number of epochs trained")
parser.add_argument("--ckpt", type=str, default="checkpoint/", help="checkpoint path")
parser.add_argument("--model_name", type=str, default="conceptual_primitives", help="model name")
parser.add_argument("--save_step", type=int, default=20000, help="number of steps to save the model")
parser.add_argument("--print_step", type=int, default=1000, help="number of steps to print the train information")
parser.add_argument("--debug_step", type=int, default=10000, help="debug step")

# set raw dataset path and glove vectors path
parser.add_argument("--ukwac_path",
                    type=str,
                    default=os.path.join(home, "utilities", "ukwac", "ukwac_pos", "pos_text"),
                    help="raw dataset")

parser.add_argument("--glove_path",
                    type=str,
                    default=os.path.join(home, "utilities", "embeddings", "glove", "glove.840B.300d.txt"),
                    help="pretrained embeddings")

# set dataset config
parser.add_argument("--save_path", type=str, default="data", help="processed dataset save path")
parser.add_argument("--dataset", type=str, default="data/dataset.txt", help="dataset file path")
parser.add_argument("--word_vocab", type=str, default="data/word_vocab.txt", help="word vocab file")
parser.add_argument("--verb_vocab", type=str, default="data/verb_vocab.txt", help="verb vocab file")
parser.add_argument("--verb_count", type=str, default="data/verb_count.txt", help="verb count file")
parser.add_argument("--word_vectors", type=str, default="data/word_vectors.npz", help="pretrained context emb")
parser.add_argument("--verb_vectors", type=str, default="data/verb_vectors.npz", help="pretrained target emb")
parser.add_argument("--word_threshold", type=int, default=100, help="word threshold")
parser.add_argument("--word_lowercase", type=boolean_string, default=True, help="word lowercase")

# for inferring
parser.add_argument("--target_verbs", type=str, default="eat,eating,ate", help="target verb for inferring")
parser.add_argument("--top_n", type=int, default=100, help="top n candidates")

dataset_size = 166000373

# parse arguments
config = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_idx)

# pre-processing dataset if not done yet
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)
    build_dataset(ukwac_path=config.ukwac_path,
                  glove_path=config.glove_path,
                  save_path=config.save_path,
                  word_threshold=config.word_threshold,
                  word_lowercase=config.word_lowercase,
                  word_dim=config.word_dim)

word_dict = read_to_dict(config.word_vocab)
verb_dict = read_to_dict(config.verb_vocab)
verb_count = read_verb_count(config.verb_count)

# create checkpoint directory
if not os.path.exists(config.ckpt):
    os.makedirs(config.ckpt)

# create model
print("building model,", end=" ", flush=True)
num_train_steps = int(dataset_size / config.batch_size) * config.epochs
model = ConceptualPrimitives(config=config,
                             verb_count=verb_count,
                             word_dict=word_dict,
                             verb_dict=verb_dict,
                             num_train_steps=num_train_steps)
print("done......", flush=True)

if config.mode == "train":
    if config.resume_training:
        model.restore_last_session()
    model.train(dataset=config.dataset, save_step=config.save_step, print_step=config.print_step)

elif config.mode == "cluster":
    # restore model
    model.restore_last_session()
    pass

elif config.mode == "infer":
    # load or extract verb related sentences from dataset
    target_verbs = config.target_verbs.lower().strip().split(",")
    verb_paths = []
    for target_verb in target_verbs:
        verb_path = extract_sentences_for_verb(target_verb, config.dataset)
        verb_paths.append(verb_path)

    # create lemmatizer
    spacy_lemmatizer = spacy.load("en", disable=['parser', 'ner'])

    # restore model
    model.restore_last_session()

    # get all possible candidates and candidate vectors
    candidate_verbs, candidate_vecs = model.get_candidates()

    # compute top candidates
    candidate_counter = Counter()
    for verb_path in verb_paths:
        total_lines = compute_num_file_lines(verb_path)
        with open(verb_path, mode="r", encoding="utf-8") as f:
            left_contexts, right_contexts, verbs = [], [], []
            cur_target = verb_path.split("/")[-1][:-4]
            for line in tqdm(f, total=total_lines, desc="compute top candidates for {}".format(cur_target)):
                line = line.lstrip().rstrip()
                if len(line) == 0:
                    continue
                if len(left_contexts) == config.batch_size:
                    candidate_list = model.inference(left_contexts, right_contexts, verbs, candidate_verbs,
                                                     candidate_vecs, top_n=config.top_n, method="multiply")
                    for candidate in candidate_list:
                        candidate_counter[candidate] += 1
                    left_contexts, right_contexts, verbs = [], [], []
                left_context, verb, right_context = line.split(SEPARATOR)
                left_contexts.append(left_context)
                right_contexts.append(right_context)
                verbs.append(verb)
            if len(left_contexts) != 0:
                candidate_list = model.inference(left_contexts, right_contexts, verbs, candidate_verbs,
                                                 candidate_vecs, top_n=config.top_n, method="multiply")
                for candidate in candidate_list:
                    candidate_counter[candidate] += 1
    # lemma
    candidate_lemma_counter = Counter()
    for candidate, count in tqdm(candidate_counter.most_common(), total=len(candidate_counter), desc="lemmatizing"):
        doc = spacy_lemmatizer(candidate)
        lemma_candidate = doc[0].lemma_
        candidate_lemma_counter[lemma_candidate] += count
    print(candidate_lemma_counter.most_common())
    candidate_save_path = os.path.join("data", "candidate")
    if not os.path.exists(candidate_save_path):
        os.makedirs(candidate_save_path)
    with open(os.path.join(candidate_save_path, "_".join(target_verbs) + ".txt"), mode="w", encoding="utf-8") as f:
        for candidate, count in candidate_lemma_counter.most_common():
            f.write("{}\t{}\n".format(candidate, count))
            f.flush()

else:
    raise ValueError("Unable to recognize mode: %s!!!" % config.mode)
