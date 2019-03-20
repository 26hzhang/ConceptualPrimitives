import os
from argparse import ArgumentParser
from models.model import ConceptualPrimitives
from utils.data_prepro import build_dataset
from models.ops_clustering import clustering
from utils.data_utils import boolean_string, read_to_dict, read_verb_count, write_csv

# set network config
parser = ArgumentParser()
parser.add_argument("--gpu_idx", type=int, nargs="+", default=[0, 1], help="GPUs used for training")
parser.add_argument("--mode", type=str, default="train", help="mode [train | cluster | infer], default, train")
parser.add_argument("--resume_train", type=boolean_string, default=False, help="resume previous trained parameters")
parser.add_argument("--neg_sample", type=int, default=10, help="number of negative samples")
parser.add_argument("--distortion", type=float, default=0.75, help="skew the unigram probability distribution")
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--num_units", type=int, default=300, help="number of units for rnn cell and hidden layer of ffn")
parser.add_argument("--k", type=int, default=100, help="number of units for output part")
parser.add_argument("--tune_emb", type=boolean_string, default=False, help="tune pretrained embeddings while training")
parser.add_argument("--use_ntn", type=boolean_string, default=True, help="if true, use neural tensor network")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--decay_step", type=int, default=int(1e4), help="learning rate decay step")
parser.add_argument("--decay_rate", type=float, default=0.9994, help="decay rate")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--epochs", type=int, default=5, help="number of epochs trained")
parser.add_argument("--ckpt", type=str, default="ckpt/", help="checkpoint path")
parser.add_argument("--max_to_keep", type=int, default=3, help="maximal checkpoints can be saved")
parser.add_argument("--model_name", type=str, default="conceptual_primitives", help="model name")
parser.add_argument("--save_step", type=int, default=10000, help="number of steps to save the model")
parser.add_argument("--print_step", type=int, default=1000, help="number of steps to print the train information")

# set raw dataset path and glove vectors path
parser.add_argument("--ukwac_path",
                    type=str,
                    default=os.path.join(os.path.expanduser("~"), "utilities", "ukwac", "ukwac_pos", "pos_text"),
                    help="raw dataset")

parser.add_argument("--glove_path",
                    type=str,
                    default=os.path.join(os.path.expanduser("~"), "utilities", "embeddings", "glove",
                                         "glove.840B.300d.txt"),
                    help="pretrained embeddings")

# set dataset config
parser.add_argument("--save_path", type=str, default="data", help="processed dataset save path")
parser.add_argument("--dataset", type=str, default="data/dataset.txt", help="dataset file path")
parser.add_argument("--word_vocab", type=str, default="data/word_vocab.txt", help="word vocab file")
parser.add_argument("--verb_vocab", type=str, default="data/verb_vocab.txt", help="verb vocab file")
parser.add_argument("--verb_count", type=str, default="data/verb_count.txt", help="verb count file")
parser.add_argument("--word_vectors", type=str, default="data/word_vectors.npz", help="pretrained context emb")
parser.add_argument("--verb_vectors", type=str, default="data/verb_vectors.npz", help="pretrained target emb")
parser.add_argument("--word_threshold", type=int, default=90, help="word threshold")
parser.add_argument("--word_lowercase", type=boolean_string, default=True, help="word lowercase")

# arguments for clustering
parser.add_argument("--emb_type", type=str, default="target", help="[init | target], default, target")
parser.add_argument("--num_cluster", type=int, default=50, help="number of clusters is required")
parser.add_argument("--cluster_method", type=str, default="kmeans", help="[kmeans | knearest]")

# parse arguments
config = parser.parse_args()
# if use ntn, then specify two GPUs for training or inferring, otherwise one is enough
gpu_str = ",".join([str(x) for x in config.gpu_idx]) if config.use_ntn else str(config.gpu_idx)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

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
model = ConceptualPrimitives(cfg=config,
                             verb_count=verb_count,
                             word_dict=word_dict,
                             verb_dict=verb_dict)

if config.mode == "train":
    if config.resume_train:
        model.restore_last_session()
    model.train(dataset=config.dataset, save_step=config.save_step, print_step=config.print_step)

elif config.mode == "cluster":
    # restore model
    model.restore_last_session()
    # extract verb embeddings and corresponding representations of the model
    vocab, vectors = model.get_verb_embs(emb_type=config.emb_type,
                                         save_path="data/{}_verb_embs.pkl".format(config.emb_type))

    # clustering
    clusters = clustering(vectors=vectors,
                          vocab=vocab,
                          num_clusters=config.num_cluster,
                          cluster_method=config.cluster_method,
                          save_path="data/{}_{}.pkl".format(config.emb_type, config.cluster_method),
                          norm=True,
                          norm_method="l2")

    # save to csv file
    write_csv(clusters, save_path="{}_{}.csv".format(config.target, config.cluster_method))

elif config.mode == "infer":
    model.restore_last_session()
    sentence = "When idle, Dave enjoys eating cake with his sister."
    verb = "eating"
    top_n = 10
    candidates = model.inference(sentence, verb, top_n=top_n, method="add", show_vec=False)
    print("Top {} candidates:".format(top_n))
    print(candidates)

else:
    raise ValueError("Unable to recognize mode: %s!!!" % config.mode)
