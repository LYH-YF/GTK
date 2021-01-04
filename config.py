import argparse
DATA_PATH = "data/Math_23K.json"
MODEL_PATH="trained_model/gtk.pkl"
PAD_TOKEN = 0
NEVER_SPLIT_TAG = []
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Training options.
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
parser.add_argument("--epochs_num",
                    type=int,
                    default=80,
                    help="Number of epochs.")
parser.add_argument("--report_steps",
                    type=int,
                    default=100,
                    help="Specific steps to print prompt.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
# Optimizer options.
parser.add_argument("--learning_rate",
                    type=float,
                    default=2e-5,
                    help="Learning rate.")
parser.add_argument("--warmup", type=float, default=0.1, help="Warm up value.")
# Model options.
parser.add_argument("--embedding_size",
                    type=int,
                    default=384,
                    help="embedding size.")
parser.add_argument("--feedforward_size",
                    type=int,
                    default=1536,
                    help="feedforward size.")
parser.add_argument("--hidden_size",
                    type=int,
                    default=384,
                    help="hidden size.")
parser.add_argument("--heads_num", type=int, default=8, help="heads num.")
parser.add_argument("--layers_num", type=int, default=12, help="layers num.")
# parser.add_argument("--",type=,default=,
#                     help=)
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--seq_length",
                    type=int,
                    default=256,
                    help="Sequence length.")
parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                "cnn", "gatedcnn", "attn", \
                                                "rcnn", "crnn", "gpt", "bilstm"], \
                                                default="bert", help="Encoder type.")
parser.add_argument("--bidirectional",
                    action="store_true",
                    help="Specific to recurrent model.")
parser.add_argument("--pooling",
                    choices=["mean", "max", "first", "last"],
                    default="first",
                    help="Pooling type.")
parser.add_argument("--vocab_size")
parser.add_argument("--input_size")
parser.add_argument("--op_nums")
parser.add_argument("--resume",type=bool,default=False,help="start train from checkpoint or a new model.")
parser.add_argument("--weight_decay", type=float, default=1e-5)
ARGS = parser.parse_args()