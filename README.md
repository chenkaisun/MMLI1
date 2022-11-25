
# Fine-Grained Chemical Entity Typing with Multimodal Knowledge Representation


## Environments
- Google Colab
- Ubuntu-18.0.4
- Python (3.7)
- Cuda (11.2)

## Installation
```
pip install -r requirements.txt
```
Install [RDKit](https://www.rdkit.org/docs/Install.html), [PyTorch](https://pytorch.org/) 1.6.0 and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) 1.7.0

## Usage

To reproduce experiment:

download files from [Google Drive](https://drive.google.com/drive/folders/1kRkJxbEZvGaec1WcyzSfnUcd7LRYHwhE?usp=sharing) and place `.pkl` and `.json` in `data_online/chemet/`, and place `.pt` file in `code/model/states/`. 

`.pkl` are preprocessed dataset including multimodal definitions, 
`cmpd_info.json` and `mention2ent.json` are external chemical definitions. `.pt` file is our saved model. The baseline/ is for ELMo Baseline in the paper.

download word vectors from [ChemPatent](https://chemu.eng.unimelb.edu.au/patent_w2v/) and place in `embeddings/`

enter `code/`

To directly produce best result for our model, run the following
```bash
python main_fet.py \
        --batch_size 6 \
        --dropout 0.2 \
        --gnn_type gine \
        --patience 8 \
        --plm_lr 5e-5 \
        --pool_type 0 \
        --lr 1e-3  \
        --model_type tdgm \
        --num_epochs 15 \
        --num_gnn_layers 1 \
        --use_cache 1 \
        --cm_type 0 \
        --exp_id 1 \
        --model_path model/states/best_dev_1.pt \
        --eval
```
---
To train Bi-LSTM, run the following
```bash
python main_fet.py  --batch_size 40 --model_name lstm --patience 8  --plm_lr 1e-3 --lr 1e-3  --num_epochs 30 --use_cache 0 --exp_id 2
```
---

To retrain other variants, note the following

``--model_type`` indicates variant type, which is combination from t(with context-only embedding), g(with molecular graph), d(with natural language description),m(with cross-modal attention), s(text-only)

the valid choices are tdgm, dgm, tdg, td, tg, s, t 

Let $MT denote the choice

run 
```bash
python main_fet.py \
        --batch_size 6 \
        --dropout 0.2 \
        --gnn_type gine \
        --patience 8 \
        --plm_lr 5e-5 \
        --pool_type 0 \
        --lr 1e-3  \
        --model_type $MT \
        --num_epochs 15 \
        --num_gnn_layers 1 \
        --use_cache 1 \
        --cm_type 0 \
        --exp_id 1 \
        --model_path model/states/best_dev.pt \
```

---
The `baseline/` is for ELMo Baseline in the paper.