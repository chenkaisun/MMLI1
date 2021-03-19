from torch.utils.data import DataLoader
import time
import torch
from torch import nn
# import wandb
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
# from model.load_model import load_model_from_path
# import logging
# from utils import *
from evaluate import evaluate
from data import collate_fn, CustomBatch, collate_wrapper
from train_utils import seed_worker
# from torch_geometric.data import DataLoader
import gc
import numpy as np
from torch.nn.functional import one_hot
import random
import numpy
from torch.utils.tensorboard import SummaryWriter
from utils import dump_file, mkdir
from IPython import embed

def train(args, model, optimizer, data):
    train_data, val_data, test_data = data

    # torch.autograd.set_detect_anomaly(True)

    if args.debug:
        pass

        # train_data = train_data.instances[:4]
        # val_data = val_data.instances[:4]
        # test_data = test_data.instances[:4]

    # collate_fn_map = {"mol_pred": collate_fn
    #
    #                   }

    if args.exp == "mol_pred":
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                  drop_last=False, num_workers=args.num_workers, worker_init_fn=seed_worker)
    else:
        print('args.exp', args.exp)
        # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_re,
        #                           drop_last=False)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_wrapper,
                                  drop_last=False)

    # model = args.model
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    # optimizer = args.optimizer

    # get logger
    logger = args.logger
    writer = args.writer

    train_iterator = range(args.start_epoch, int(args.num_epochs) + args.start_epoch)
    total_steps = int(len(train_loader) * args.num_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    # scheduler = CosineAnnealingLR(optimizer, T_max=(int(args.num_epochs) // 4) + 1, eta_min=0)

    logger.debug(f"Total steps: {total_steps}")
    logger.debug(f"Warmup steps: {warmup_steps}")

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    bad_counter = 0
    best_val_score = -float("inf")
    best_epoch = 0
    t_total = time.time()
    num_steps = 0
    logger.debug(f"{len(train_loader)} steps for each epoch")
    for epoch in train_iterator:
        logger.debug(f"Epoch {epoch}")
        t = time.time()

        # torch.autograd.set_detect_anomaly(True)

        total_loss = 0
        for step, batch in enumerate(train_loader):
            # logger.debug(f"Step {step}")
            num_steps += 1

            if args.exp == "mol_pred":

                encoded_input = batch[0]  # tokenizer(batch[0])
                # print("encoded_input", encoded_input)
                inputs = {'input_ids': {key: encoded_input[key].to(args.device) for key in encoded_input},
                          'batch_graph_data': batch[1].to(args.device),
                          'ids': batch[2],
                          'in_train': True,
                          }

            else:
                # texts = batch[0]
                # batch_ent1_d = batch[1]
                # batch_ent2_d = batch[3]
                #
                # # lb = batch[10].int().numpy()
                # # tmp=np.zeros((len(lb), 13))
                # #
                # # tmp[np.arange(len(lb)), lb] = 1
                # # tmp=torch.tensor(tmp)
                # # print("encoded_input", encoded_input)
                # inputs = {'texts': {key: texts[key].to(args.device) for key in texts},
                #           "batch_ent1_d": {key: batch_ent1_d[key].to(args.device) for key in batch_ent1_d},
                #           "batch_ent1_d_mask": batch[2].to(args.device),
                #           "batch_ent2_d": {key: batch_ent2_d[key].to(args.device) for key in batch_ent2_d},
                #           "batch_ent2_d_mask": batch[4].to(args.device),
                #           "batch_ent1_g": batch[5].to(args.device),
                #           "batch_ent1_g_mask": batch[6].to(args.device),
                #           "batch_ent2_g": batch[7].to(args.device),
                #           "batch_ent2_g_mask": batch[8].to(args.device),
                #           "ids": batch[9],
                #           # "labels": tmp.to(args.device),
                #           "labels": batch[10].to(args.device),
                #           'in_train': True,
                #           }
                inputs = batch.to(args.device)
            # inputs = {'input_ids': {key:encoded_input[key].to(args.device) for key in encoded_input},
            #           'edge_indices': batch[1].to(args.device),
            #           'node_attrs': batch[2].to(args.device),
            #           'edge_attrs': batch[3].to(args.device),
            #           'Ys': batch[4].to(args.device),
            #           'ids': batch[5],
            #           'in_train': True,
            #           }

            # model learning
            model.train()

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    loss = model(inputs, args)
                scaler.scale(loss).backward()

                if (step + 1) % args.grad_accumulation_steps == 0 or step == len(train_loader) - 1:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                loss = model(inputs, args)
                loss.backward()
                if step % args.grad_accumulation_steps == 0 or step == len(train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    # scheduler.step()
                    optimizer.zero_grad()
            total_loss += loss.item()
            # else:
            #     loss = model(inputs)
            #     loss.backward()
            #     if step % args.grad_accumulation_steps == 0 or step == len(train_loader) - 1:
            #         optimizer.step()
            #         scheduler.step()
            #         optimizer.zero_grad()

            # if args.n_gpu > 1:
            #     loss = loss.mean()

            # if args.max_grad_norm > 0:
            #     if args.amp:
            #         scaler.unscale_(optimizer)
            #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            # else:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # wandb.log({"loss": loss.item()}, step=num_steps)

        val_score, output = evaluate(args, model, val_data)

        # mkdir("analyze")
        # dump_file(output, "analyze/output.json")

        if epoch > args.burn_in:
            if val_score >= best_val_score:
                best_val_score, best_epoch, bad_counter = val_score, epoch, 0
                torch.save({
                    'epoch': epoch + 1,
                    'num_steps': num_steps + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_score': best_val_score,
                }, args.model_path)
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break

        logger.debug(f'Epoch {epoch} | Train Loss {total_loss:.8f} | Val Score {val_score:.4f} | '
                     f'Time Passed {time.time() - t:.4f}s')
        # embed()

        writer.add_scalar('train', total_loss, epoch)
        writer.add_scalar('val', val_score, epoch)

        # wandb.log({'loss_train': loss.data.item(),
        #            'val_score': val_score,
        #            }, step=num_steps)
        # print('Time passed', (time.time() - t))
    logger.debug("Optimization Finished!")
    logger.debug("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    logger.debug('Loading {}th epoch'.format(best_epoch))

    gc.collect()
    model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    test_score, output = evaluate(args, model, test_data)

    # mkdir("analyze")
    # dump_file(output, "analyze/output.json")

    logger.debug(f"Test Score {test_score}")
    writer.add_scalar('test', test_score, 0)
    writer.add_hparams(
        {'batch_size': args.batch_size, 'num_epochs': args.num_epochs,
         'plm_lr': args.plm_lr, 'lr': args.lr, 'g_dim': args.g_dim, 'max_grad_norm': args.max_grad_norm,
         'mult_mask': args.mult_mask, 'g_mult_mask': args.g_mult_mask, 'dropout': args.dropout, 'model_type': args.model_type,
          'g_global_pooling': args.g_global_pooling},
        {'hparam/test': test_score, 'hparam/val': best_val_score})
    writer.close()
