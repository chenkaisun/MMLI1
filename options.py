import argparse


def read_args():
    parser = argparse.ArgumentParser()

    # pretrained language model
    parser.add_argument("--plm", default="bert-base-cased", type=str, metavar='N')
    parser.add_argument("--max_seq_len", default=1024, type=int)

    # model
    parser.add_argument("--model_name", default="", type=str)
    parser.add_argument("--model_path", default="model/states/best_dev.pt", type=str)
    parser.add_argument("--experiment", default="", type=str)
    parser.add_argument("--experiment_path", default="experiment/", type=str)

    # data
    parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--val_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--cache_data", default=True, type=str)

    # training params
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--plm_lr", default=5e-5, type=float, help="The initial learning rate for PLM.")
    parser.add_argument("--lr", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam.")
    parser.add_argument("--num_epochs", default=300, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--eval_epoch", default=30, type=float, help="Number of steps between each evaluation.")
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--burn_in', type=int, default=20)
    parser.add_argument('--print_epoch_interval', type=int, default=10)
    parser.add_argument("--scheduler", default="")
    parser.add_argument("--n_gpu", default=1, type=int, help="Number of gpu", )
    parser.add_argument("--use_gpu", default=1, type=int, help="Using gpu or cpu", )
    parser.add_argument("--use_amp", default=1, type=int, help="Using mixed precision")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int, help="Using mixed precision")

    # model params
    parser.add_argument("--dropout", default=0.3, type=float, help="Dropout")
    parser.add_argument('--hidden_dim', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--embedding_dim', type=int, default=16, help='Number of embedding units.')
    parser.add_argument('--batch_norm', default=False, help="Please give a value for batch_norm")

    # auxiliary
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()
    return args
