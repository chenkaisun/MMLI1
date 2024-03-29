import argparse


def read_args():
    parser = argparse.ArgumentParser()

    # pretrained language model
    parser.add_argument("--plm", default="bert-base-cased", type=str, metavar='N')
    parser.add_argument("--max_seq_len", default=1024, type=int)

    # experiment
    parser.add_argument("--model_name", default="fet_model", type=str)
    parser.add_argument("--model_path", default="model/states/best_dev.pt", type=str)
    parser.add_argument("--experiment", default="exp", type=str)
    parser.add_argument("--experiment_path", default="../experiment/", type=str)
    parser.add_argument("--exp", default="fet", type=str)
    parser.add_argument("--exp_id", default="0", type=str)
    parser.add_argument("--analyze", default=0, type=int)
    parser.add_argument("--add_concept", default=0, type=int)
    parser.add_argument("--add_label_text", default=0, type=int)

    # data
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--tgt_name", default="p_np", type=str)

    parser.add_argument("--data_dir", default="data/property_pred/clintox.csv", type=str)

    parser.add_argument("--train_file", default="data/property_pred/clintox.csv", type=str)
    parser.add_argument("--val_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--cache_filename", default="", type=str)
    parser.add_argument("--use_cache", default=0, type=int)
    parser.add_argument("--cache_data", default=0, type=int)

    # training params
    parser.add_argument("--num_atom_types", default=0, type=int)
    parser.add_argument("--num_edge_types", default=0, type=int)
    parser.add_argument("--batch_size", default=6, type=int, help="Batch size for training.")
    parser.add_argument("--plm_lr", default=2e-5, type=float, help="The initial learning rate for PLM.")
    parser.add_argument("--lr", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--activation", default="gelu", type=str)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam.")
    parser.add_argument("--num_epochs", default=15, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--eval_epoch", default=30, type=float, help="Number of steps between each evaluation.")
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--burn_in', type=int, default=0)
    parser.add_argument('--print_epoch_interval', type=int, default=10)
    parser.add_argument("--scheduler", default=1)
    # parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag', required=True)

    parser.add_argument("--gpu_id", default=0, type=int, help="gpu_id", )
    parser.add_argument("--n_gpu", default=1, type=int, help="Number of gpu", )
    parser.add_argument("--use_gpu", default=1, type=int, help="Using gpu or cpu", )
    parser.add_argument("--use_amp", default=1, type=int, help="Using mixed precision")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int, help="Using mixed precision")
    parser.add_argument("--num_workers", default=1, type=int)

    # model params
    parser.add_argument("--in_dim", default=14, type=float, help="Feature dim")
    parser.add_argument("--out_dim", default=14, type=float, help="Feature dim")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout")

    parser.add_argument('--g_dim', type=int, default=256, help='Number of final hidden units for graph.')
    parser.add_argument('--num_gnn_layers', type=int, default=2, help='Number of final hidden units for graph.')

    parser.add_argument('--plm_hidden_dim', type=int, default=768, help='Number of hidden units for plm.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--embedding_dim', type=int, default=16, help='Number of embedding units.')
    parser.add_argument('--batch_norm', default=False, help="Please give a value for batch_norm")
    # parser.add_argument('--i_only', default=0, type=int)
    # parser.add_argument('--g_only', default=0, type=int)
    # parser.add_argument('--t_only', default=0, type=int)
    # parser.add_argument('--i', default=0, type=int)
    # parser.add_argument('--gt', default=0, type=int)
    # parser.add_argument('--t', default=0, type=int)
    # parser.add_argument('--td', default=0, type=int)
    # parser.add_argument('--tg', default=0, type=int)
    # parser.add_argument('--tdg', default=0, type=int)
    # parser.add_argument('--tdg_x', default=0, type=int)
    parser.add_argument('--model_type', default="tdgm")
    parser.add_argument('--mult_mask', default=0, type=int)
    parser.add_argument('--g_mult_mask', default=0, type=int)
    parser.add_argument('--g_global_pooling', default=1, type=int)
    parser.add_argument('--gnn_type', default="gine")
    parser.add_argument('--cm_type', default=0, type=int)  # 0 original, 1 no tformer, 2 3D
    parser.add_argument('--pool_type', default=0, type=int)  # for cm, 0 mean max, 1 max mean, 2 mean, 3 max
    parser.add_argument('--type_embed', default=0, type=int)
    parser.add_argument('--cm', default=0, type=int)
    parser.add_argument('--attn_analysis', default=0, type=int)
    parser.add_argument('--error_analysis', default=0, type=int)


    ##for lst

    parser.add_argument('--embed_dim', type=int, default="200", help='Number of embedding units.')
    parser.add_argument('--word_embed_type', type=int, default=16, help='Number of embedding units.')
    parser.add_argument('--lstm_dropout', type=int, default=.5)
    parser.add_argument('--embed_dropout', type=int, default=.5)

    # auxiliary
    # parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval", action="store_true")

    # # experiment specific
    # parser.add_argument("--g", action="store_true")
    # parser.add_argument("--tg", action="store_true")

    args = parser.parse_args()
    return args
