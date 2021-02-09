from torch.utils.data import DataLoader
import time
import torch
from torch import nn
import wandb
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
# from model.load_model import load_model_from_path
# import logging
# from utils import *
from evaluate import evaluate
from data import collate_fn
from utils import get_logger
# from torch_geometric.data import DataLoader
import gc

def train(args, model, optimizer, data):
    train_data, val_data, test_data = data

    if args.debug:
        pass
        # train_data = train_data[:500]
        # val_data = train_data
        # test_data = train_data
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              drop_last=False)
    # model = args.model
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    # optimizer = args.optimizer

    # get logger
    logger = get_logger(args)

    train_iterator = range(args.start_epoch, int(args.num_epoch) + args.start_epoch)
    total_steps = int(len(train_loader) * args.num_epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
    #                                             num_training_steps=total_steps)
    # scheduler = CosineAnnealingLR(optimizer, T_max=(int(args.num_epoch) // 4) + 1, eta_min=0)

    logger.debug(f"Total steps: {total_steps}")
    logger.debug(f"Warmup steps: {warmup_steps}")

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    bad_counter = 0
    best_val_score = -float("inf")
    best_epoch = 0
    t_total = time.time()
    num_steps = 0
    for epoch in train_iterator:
        logger.debug(f"Epoch {epoch}")
        t = time.time()

        # torch.autograd.set_detect_anomaly(True)

        total_loss=0
        for step, batch in enumerate(train_loader):
            logger.debug(f"Step {step}")
            num_steps += 1
            encoded_input = batch[0]  # tokenizer(batch[0])
            # print("encoded_input", encoded_input)
            inputs = {'input_ids': {key: encoded_input[key].to(args.device) for key in encoded_input},
                      'batch_graph_data': batch[1].to(args.device),
                      'ids': batch[2],
                      'in_train': True,
                      }

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
                    loss = model(inputs)
                scaler.scale(loss).backward()

                if step % args.grad_accumulation_steps == 0 or step == len(train_loader) - 1:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    # scheduler.step()
                    optimizer.zero_grad()
            else:
                loss = model(inputs)
                loss.backward()
                if step % args.grad_accumulation_steps == 0 or step == len(train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    # scheduler.step()
                    optimizer.zero_grad()
            total_loss+=loss.item()
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
    logger.debug(f"Test Score {test_score}")
