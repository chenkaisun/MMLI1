
# Fine-Grained Chemical Entity Typing with Multimodal Knowledge Representation


## Environments
- Google Colab
- Ubuntu-18.0.4
- Python (3.7)
- Cuda (10.1)

## Installation
```
pip install -r requirements.txt
```
Install [RDKit](https://www.rdkit.org/docs/Install.html), [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
  
## Usage

To reproduce experiment:

download files from [Google Drive](https://drive.google.com/drive/folders/1kRkJxbEZvGaec1WcyzSfnUcd7LRYHwhE?usp=sharing) and place `.pkl` and `.json` in `data_online/chemet/`, and place `.pt` file in `code/model/states/`


`.pkl` are preprocessed dataset including multimodal definitions, 
`cmpd_info.json` and `mention2ent.json` are external chemical definitions

Then go into `code/` and run the following
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

