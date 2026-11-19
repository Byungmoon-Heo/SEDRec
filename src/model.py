import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import copy
import numpy as np
from step_sample import LossAwareSampler
import torch as th
import einops
import os
from common import *
from sedrec import SEDRec

class Att_Diffuse_model(nn.Module):
    def __init__(self, args):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.args=args
        self.item_num = args.item_num
        self.item_embedding = self.embed_item(pretrained=args.pretrained)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.hist_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = create_model_diffu(args)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=0)
        self.geodesic = args.geodesic
    def load_pretrained_emb_weight(self):

        path = os.path.join('saved','pretrain',self.args.dataset, 'pretrain.pth')
        saved = torch.load(path, map_location='cpu',weights_only=False)
        pretrained_emb_weight = saved['item_embedding.weight']
        return pretrained_emb_weight
    def embed_item(self,pretrained=False):
        if pretrained:
            embedding = nn.Embedding.from_pretrained(
                self.load_pretrained_emb_weight(), padding_idx=0, freeze=self.args.freeze_emb
            )
        else:
            embedding = nn.Embedding(self.item_num+1, self.emb_dim, padding_idx=0)
        return embedding


    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_diffu(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embedding.weight.t())
        scores_pos = scores.gather(1 , labels)  
        scores_neg_mean = (torch.sum(scores, dim=-1).unsqueeze(-1)-scores_pos)/(scores.shape[1]-1)

        loss = torch.min(-torch.log(torch.mean(torch.sigmoid((scores_pos - scores_neg_mean).squeeze(-1)))), torch.tensor(1e8))
       
        return loss

    def calculate_loss_minibatch(self, out_seq, labels, batch_size=128):
        item_embeddings = self.item_embedding.weight.t()
        num_batches = out_seq.shape[0] 
        total_loss = 0.0
        num = num_batches//batch_size
        
        for i in range(0, num_batches, batch_size):
            batch_out_seq = out_seq[i:i + batch_size] 
            batch_labels = labels[i:i + batch_size] 
            scores = torch.matmul(batch_out_seq, item_embeddings)
            loss = self.loss_ce(scores.reshape(-1, scores.shape[-1]), batch_labels.reshape(-1))
            total_loss += loss
        return total_loss / num

    def calculate_loss(self, out_seq, labels):
        index = labels>0
        out_seq = out_seq[index]
        labels = labels[index]
        scores = torch.matmul(out_seq, self.item_embedding.weight.t()) #B,L,K
        loss = self.loss_ce(scores.reshape(-1, scores.shape[-1]), labels.reshape(-1))
        return loss
        
    def calculate_score(self, item):
        scores = torch.matmul(item.reshape(-1, item.shape[-1]), self.item_embedding.weight.t())
        return scores
    
    def loss_rmse(self, rep_diffu, labels):
        rep_gt = self.item_embedding(labels).squeeze(1)
        return torch.sqrt(self.loss_mse(rep_gt, rep_diffu))

    def forward(self, sequence, tag, train_flag=True):
        item_embeddings = self.item_embedding(sequence)
        tag_embeddings = self.item_embedding(tag)
        if self.geodesic:
            tag_embeddings = F.normalize(tag_embeddings,p=2, dim=-1)
        item_embeddings = self.embed_dropout(item_embeddings)  
        item_embeddings = self.hist_norm(item_embeddings)
        mask_seq = (sequence>0).float()
        mask_tag = (tag>0).float().view(tag.shape[0],-1)
        
        if train_flag:
            if self.args.model == "sedrec":
                out_seq, dif_loss = self.diffu(
                    item_embeddings, tag_embeddings, mask_seq, mask_tag,
                    tag_ids=tag,       
                    seq_ids=sequence   
                  )
            else:
                out_seq, dif_loss = self.diffu(item_embeddings, tag_embeddings, mask_seq, mask_tag)

            last_item = out_seq[:, -1, :]


        else:
            if self.args.model == "sedrec":
                out_seq = self.diffu.denoise_sample(
                    item_embeddings, tag_embeddings, mask_seq, mask_tag,
                    seq_ids=sequence
                )
            else:
                out_seq = self.diffu.denoise_sample(item_embeddings, tag_embeddings, mask_seq, mask_tag)

            last_item = out_seq[:, -1, :]
            dif_loss = None
        return out_seq, last_item, dif_loss


def create_model_diffu(args):
    if args.model == 'sedrec':
        return SEDRec(args)
    else:
        print('args.model is wrong')
        return None
