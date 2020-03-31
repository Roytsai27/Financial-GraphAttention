import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run GAT.")
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--channel_size', type=int, default=16,
                        help='Number of channel_size.')
    parser.add_argument('--dim', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--l2', type=float, default=0,
                        help='Regularization for embeddings.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='loss weight for ranking loss.')
    parser.add_argument('--device', type=str, default="cuda:1",
                        help='Device id')
    return parser.parse_args()


def parse_basic_args():
    parser = argparse.ArgumentParser(description="Run GAT.")
    parser.add_argument('--data', type=str, default="Taiwan_model_data_10_best.pickle",
                        help='Data path.')
    parser.add_argument('--model', type=str, default="CAT",
                        help='Model for training, choose from [CG ,CAT,CPool].')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--dual_attention', type=bool, default=False,
                        help='Whether apply two attention separatly for cls and reg.')
    parser.add_argument('--dim', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--l2', type=float, default=0,
                        help='Regularization for embeddings.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--alpha', type=float, default=1,
                        help='loss weight for mae loss.')
    parser.add_argument('--beta', type=float, default=1,
                        help='loss weight for cls loss.')
    parser.add_argument('--gamma', type=float, default=1,
                        help='loss weight for rank loss.')
    parser.add_argument('--device', type=str, default="cuda:1",
                        help='Device id')
    parser.add_argument('--use_gru', type=bool, default=False,
                        help='Whther use gru')
    parser.add_argument('--week_num', type=int, default=3,
                        help='Number of weeks')
    parser.add_argument('--weight', type=float, default=0.5,
                        help='Classification threshold')
    return parser.parse_args()


def parse_rank_lstm_args():
    parser = argparse.ArgumentParser(description="Run GAT.")
    parser.add_argument('--data', type=str, default="Taiwan_model_data_10_best.pickle",
                        help='Data path.')
    parser.add_argument('--model', type=str, default="Taiwan",
                        help='Model for training, choose from [Taiwan, SP500].')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--dim', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--l2', type=float, default=0,
                        help='Regularization for embeddings.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--device', type=str, default="cuda:1",
                        help='Device id')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args.epochs)