# SEDRec
Official source code for CIKM 2026 paper "Semantic Entanglement for Generative Diffusion-based Sequential Recommendation"
<br/>

## Environment Setting:

- numpy==1.24.4
- pandas==2.2.2
- python==3.10.12
- torch==2.5.0
- transformer==4.57.6
- sentence-transformer==3.4.1
<br>


## Dataset Preparation
1. Run the following command.
```bash
python prepare_data.py \
  --root ./ \
  --category Beauty \
  --diffurec_out_dir ./prepare_datasets/beauty \
  --metadata sentence \
  --emb_backend sentence-transformers \
  --emb_model sentence-transformers/sentence-t5-base \
  --emb_batch_size 256 \
  --device cuda:0
```

2. Once you have generated semantic-embedding.npy and dataset.pkl, move these files to `datasets/data/sports|beauty|toys`.

<br/>

If you want to download the dataset directly, please use the link.
[[Google Drive](https://drive.google.com/drive/folders/1jpUlLSQOCqTViM9hHKFANUVVHBBX6jXj?usp=sharing)]
<br/>
After downloading, place the file in the `datasets/data/sports|beauty|toys` folder.


## Pretrain Embedding
```bash
python main.py --dataset sports --model pretrain
```

## Training
```bash
python main.py --dataset sports --model sedrec
```

## Citation:
If this work is helpful for you, please cite our paper:
```bibtex
@inproceedings{heo2026sedrec,
  title={SEDRec: Semantic Entanglement for Generative Diffusion-based Sequential Recommendation},
  author={Heo, Byungmoon and Kim, Jaekwang},
  booktitle={Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  pages={1560--1570},
  year={2026}
}
```

## Acknowledgments
This repository is based on [ADRec](https://github.com/Nemo-1024/ADRec).
For semantic embedding extraction, genrec folder is from [RPG](https://github.com/facebookresearch/RPG_KDD2025).
