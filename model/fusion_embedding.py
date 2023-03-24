from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertEmbeddings


class PinyinCNNEmbedding(nn.Module):
    def __init__(self, embedding_size: int, pinyin_out_dim: int):
        """
            Pinyin Embedding Module
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """

        super(PinyinCNNEmbedding, self).__init__()
        self.pinyin_out_dim = pinyin_out_dim
        self.embedding = nn.Embedding(1109, embedding_size)
        with open("vectors.txt", 'r', encoding='utf-8') as pf:
            self.pinyin_embedding = []
            for line in pf.readlines():
                line = line.strip("\r\n").split(" ")
                data = [float(item) for item in line[1:]]
                self.pinyin_embedding.append(data)
        self.embedding.from_pretrained(torch.from_numpy(np.array(self.pinyin_embedding)))
        self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.pinyin_out_dim, kernel_size=2,
                              stride=1, padding=0)
        self.init_weight()

    def init_weight(self):
        self.embedding.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(self.conv.weight.data,
                                mode='fan_out',
                                nonlinearity='relu')
    def forward(self, pinyin_ids):
        embed = self.embedding(pinyin_ids)
        return embed
    # def forward(self, pinyin_ids):
    #     """
    #     Args:
    #         pinyin_ids: (bs*sentence_length*pinyin_locs)
    #     Returns:
    #         pinyin_embed: (bs,sentence_length,pinyin_out_dim)
    #     """
    #     # input pinyin ids for 1-D conv
    #     embed = self.embedding(pinyin_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
    #     bs, sentence_length, pinyin_locs, embed_size = embed.shape
    #     view_embed = embed.view(-1, pinyin_locs, embed_size)  # [(bs*sentence_length),pinyin_locs,embed_size]
    #     input_embed = view_embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, pinyin_locs]
    #     # conv + max_pooling
    #     pinyin_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
    #     pinyin_embed = F.max_pool1d(pinyin_conv, pinyin_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
    #     return pinyin_embed.view(bs, sentence_length, self.pinyin_out_dim)  # [bs,sentence_length,pinyin_out_dim]


class FusionBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.pinyin_embeddings = PinyinCNNEmbedding(embedding_size=128, pinyin_out_dim=config.hidden_size)
        self.map_fc = nn.Linear(config.hidden_size+128, config.hidden_size)
        self.init_weight()

    def init_weight(self):
        self.map_fc.weight.data.normal_(0.0, 0.02)
        self.map_fc.bias.data.zero_()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pinyin_ids: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        ##
        word_embeddings = inputs_embeds  # [bs,l,hidden_size]
        pinyin_embeddings = self.pinyin_embeddings(pinyin_ids)  # [bs,l,hidden_size]
        # word_embeddings = F.normalize(word_embeddings, dim=2)
        # pinyin_embeddings = F.normalize(pinyin_embeddings, dim=2)
        # fusion layer
        concat_embeddings = torch.cat((word_embeddings, pinyin_embeddings), 2)
        # concat_embeddings = word_embeddings + pinyin_embeddings
        inputs_embeds = self.map_fc(concat_embeddings)
        ##

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
