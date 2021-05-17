# from utils import *
#
# output=load_file("analyze/output.json")

import torch
from transformers import BertConfig, BertTokenizer, BertModel

model_type = 'bert-base-uncased'
model_type = "prajjwal1/bert-tiny"
model_type = "prajjwal1/bert-tiny"
config = BertConfig.from_pretrained(model_type)
config.output_attentions = True
model = BertModel.from_pretrained(model_type, config=config).to('cuda')
tokenizer = BertTokenizer.from_pretrained(model_type)

def get_attn(model, tokenizer, ):
    text1 = 'We met today and she wanted to'
    text2 = 'meet again'
    tok1 = tokenizer.tokenize(text1)
    tok2 = tokenizer.tokenize(text2)

    p_pos = len(tok1)  # position for token
    tok = tok1 + tok2
    tok, p_pos, tok[p_pos]

    ids = torch.tensor(tokenizer.convert_tokens_to_ids(tok)).unsqueeze(0).to('cuda')
    with torch.no_grad():
        output = model(ids)
    attentions = torch.cat(output[2]).to('cpu')
    attentions.shape #(layer, batch_size (squeezed by torch.cat), num_heads, sequence_length, sequence_length)

    attentions = attentions.permute(2, 1, 0, 3)
    print(attentions.shape)  # (sequence_length, num_heads, layer, sequence_length)

    layers = len(attentions[0][0])
    heads = len(attentions[0])
    seqlen = len(attentions)
    layers, heads, seqlen

    attentions = attentions.permute(2, 1, 0, 3)


    attentions_pos = attentions[1]
    attentions_pos.shape

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    cols = 2
    rows = int(heads / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5))
    axes = axes.flat
    print(f'Attention weights for token {tok[p_pos]}')
    for i, att in enumerate(attentions_pos):
        # print(att)
        print(att.shape)
        # im = axes[i].imshow(att, cmap='gray')
        ax = sns.heatmap(att, vmin=0, vmax=1, ax=axes[i], xticklabels=tok, yticklabels=tok)
        ax.xaxis.tick_top()  # x axis on top
        ax.xaxis.set_label_position('top')
        plt.xticks(rotation=90)
        axes[i].set_title(f'head - {i} ')
        axes[i].set_ylabel('layers')

    plt.savefig("../paper/attn.png")
    plt.show()
