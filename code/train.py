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
# import numpy
# from torch.utils.tensorboard import SummaryWriter
from utils import dump_file, mkdir
from IPython import embed
import os
from utils import load_file, dump_file, visualize_plot
import gc

# from torch.optim.lr_scheduler import _LRScheduler

def train(args, model, optimizer, data):
    train_data, val_data, test_data = data

    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        num_samples_test=20
        train_data.instances = train_data.instances[:num_samples_test]
        val_data.instances = val_data.instances[:num_samples_test]
        test_data.instances = test_data.instances[:num_samples_test]

    # embed()
    if args.exp == "mol_pred":
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                  drop_last=False, num_workers=args.num_workers, worker_init_fn=seed_worker)
    else:
        print('args.exp', args.exp)
        # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_re,
        #                           drop_last=False)
        #shuffle now true
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_wrapper,
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

    scheduler = None
    if args.scheduler:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

        # scheduler = STLR(optimizer, num_warmup_steps=warmup_steps,
        #                                             num_training_steps=total_steps)

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
    # print("train_iterator",train_iterator)
    for epoch in train_iterator:
        gc.collect()
        # logger.debug(f"Epoch {epoch}")
        t = time.time()

        # torch.autograd.set_detect_anomaly(True)

        total_loss = 0
        for step, batch in enumerate(train_loader):
            # logger.debug(f"Step {step}")
            # gc.collect()

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
                inputs = batch.to(args.device)
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
                    if args.scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
            else:
                loss = model(inputs, args)
                loss.backward()
                if step % args.grad_accumulation_steps == 0 or step == len(train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if args.scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
            total_loss += loss.item()

        val_score, output = evaluate(args, model, val_data)

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
    # writer.add_scalar('test', test_score, 0)
    # writer.add_hparams(
    #     {'batch_size': args.batch_size, 'num_epochs': args.num_epochs,
    #      'plm_lr': args.plm_lr, 'lr': args.lr, 'g_dim': args.g_dim, 'max_grad_norm': args.max_grad_norm,
    #      'mult_mask': args.mult_mask, 'g_mult_mask': args.g_mult_mask, 'dropout': args.dropout, 'model_type': args.model_type,
    #       'g_global_pooling': args.g_global_pooling},
    #     {'hparam/test': test_score, 'hparam/val': best_val_score})
    # writer.close()

    sr_file = args.experiment_path + args.exp + "_result.json"
    sr = load_file(sr_file) if os.path.exists(sr_file) else []
    hparam = vars(args)

    # print("e2")
    # serialize params
    for key in hparam:


        item=hparam[key]
        if not isinstance(item, (float, str, int, complex, list, dict, set, frozenset, bool)):
            hparam[key]=str(item)
    hparam["val_score"] = best_val_score
    hparam["test_score"] = test_score
    sr.append(hparam)

    # print("e0")

    # Plot lines
    visualize_plot(y=[[hparam["val_score"] for hparam in sr],
                      [hparam["test_score"] for hparam in sr]],
                   name=["val", "test"],
                    path=args.experiment_path + args.exp + "_result.png")

    # print("e1")
    dump_file(sr, sr_file)

# class STLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, max_mul, ratio, steps_per_cycle, decay=1, last_epoch=-1):
#         self.max_mul = max_mul - 1
#         self.turning_point = steps_per_cycle // (ratio + 1)
#         self.steps_per_cycle = steps_per_cycle
#         self.decay = decay
#         super().__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         residual = self.last_epoch % self.steps_per_cycle
#         multiplier = self.decay ** (self.last_epoch // self.steps_per_cycle)
#         if residual <= self.turning_point:
#             multiplier *= self.max_mul * (residual / self.turning_point)
#         else:
#             multiplier *= self.max_mul * (
#                 (self.steps_per_cycle - residual) /
#                 (self.steps_per_cycle - self.turning_point))
#         return [lr * (1 + multiplier) for lr in self.base_lrs]
#
# class NoamLR(_LRScheduler):
#     """
#     Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
#     linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
#     to the inverse square root of the step number, scaled by the inverse square root of the
#     dimensionality of the model. Time will tell if this is just madness or it's actually important.
#     Parameters
#     ----------
#     warmup_steps: ``int``, required.
#         The number of steps to linearly increase the learning rate.
#     """
#     def __init__(self, optimizer, warmup_steps):
#         self.warmup_steps = warmup_steps
#         super().__init__(optimizer)
#
#     def get_lr(self):
#         last_epoch = max(1, self.last_epoch)
#         scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
#         return [base_lr * scale for base_lr in self.base_lrs]
