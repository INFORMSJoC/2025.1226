import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from typing import Optional
from dl.diffpool.model.encoder import DiffPool as DiffPool


def create_look_ahead_mask(size):
    mask = torch.tril(torch.ones(size, size) == 1).float()
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


def create_padding_mask(data):
    return data[:, :, -1] == 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_len, device=None):
        super(PositionalEncoding, self).__init__()
        pos_encoding = torch.zeros(max_len, d_embed)
        pos_encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding = pos_encoding.unsqueeze(0).to(device)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.pos_encoding[:, :seq_len, :]
        return x + pos_embed


class Tran2SP(nn.Module):
    def __init__(
        self,
        src_dim,
        tgt_dim,
        scenario_dim,
        num_gnn_layers,
        n_pooling,
        assign_dim,
        pool_ratio,
        dp_hidden_dim,
        dp_dropout,
        g_out="raw",
        d_model=512,
        nhead=8,
        num_decoder_layer=6,
        dim_feedforward=2048,
        scenario_embed_dim1=256,
        scenario_embed_dim2=256,
        dropout=0.1,
        dropout_embed=0.1,
        pos_encoding_max_len=256,
        activation=F.relu,
        edge_dim=1,
        custom_decoder=None,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=False,
        training=True,
        device=None,
        dtype=None,
    ):
        super(Tran2SP, self).__init__()
        self.name = "Tran2SP"
        self.dropout_embed = dropout_embed
        self.device = device
        self.training = training

        diffpool_kwargs = dict(
            input_dim=src_dim,
            hidden_dim=dp_hidden_dim,
            embedding_dim=d_model,
            activation=F.relu,
            n_layers=num_gnn_layers,
            dropout=dp_dropout,
            n_pooling=n_pooling,
            linkpred=False,
            aggregator_type="meanpool",
            assign_dim=assign_dim,
            pool_ratio=pool_ratio,
            out_type=g_out,
            edge_dim=edge_dim
        )
        self.feature_embed = DiffPool(**diffpool_kwargs)

        self.linear_tgt = nn.Linear(tgt_dim - 1 + 4, d_model)

        self.positional_encoding = PositionalEncoding(d_embed=d_model, max_len=pos_encoding_max_len, device=device)

        self.embed_layer1 = nn.Sequential(nn.Linear(scenario_dim, scenario_embed_dim1), nn.ReLU())
        self.embed_layer2 = nn.Sequential(nn.Linear(scenario_embed_dim1, scenario_embed_dim2), nn.ReLU())
        self.embed_layer3 = nn.Sequential(nn.Linear(scenario_embed_dim2, d_model))


        if not custom_decoder:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
                device=device,
                dtype=dtype,
            )
            decoder_norm = nn.LayerNorm(
                d_model,
                eps=layer_norm_eps,
                device=device,
                dtype=dtype,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=num_decoder_layer, norm=decoder_norm
            )
        else:
            self.decoder = custom_decoder

        self.linear_out_cut = nn.Linear(d_model, tgt_dim - 1)
        self.linear_out_token = nn.Linear(d_model, 4)

    def forward(
        self,
        tgt,
        context_graph,
        scenario,
        n_scen=None,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        src = self.feature_embed(context_graph)

        scen_embed = self.embedding_scenario(scenario, n_scen)
        src = torch.concat((src, scen_embed), dim=1)

        token_encoding = F.one_hot(tgt[:, :, -1].to(torch.long), num_classes=4)
        tgt = torch.concat((tgt[:, :, :-1], token_encoding), dim=-1)
        tgt = self.linear_tgt(tgt)
        tgt = self.positional_encoding(tgt)

        out = self.decoder(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        out_cut = self.linear_out_cut(out)
        out_token = torch.softmax(self.linear_out_token(out), dim=-1)

        return torch.concat((out_cut, out_token), dim=-1)

    def embedding_scenario(self, scenario, n_scen):
        if n_scen is None:
            embed = self.embed_layer1(scenario.float())
            if self.dropout_embed:
                embed = F.dropout(embed, self.dropout_embed, training=self.training)

            embed = self.embed_layer2(embed)
            if self.dropout_embed:
                embed = F.dropout(embed, self.dropout_embed, training=self.training)

            embed = torch.mean(embed, dim=1, keepdim=True)
            embed = self.embed_layer3(embed)
            if self.dropout_embed:
                scen_embed = F.dropout(embed, self.dropout_embed, training=self.training)
            else:
                scen_embed = embed
        else:
            scen_embed = []
            for i, scen in enumerate(scenario):
                n = n_scen[i]
                embed = scen[:n]
                embed = self.embed_layer1(embed.float())
                if self.dropout_embed:
                    embed = F.dropout(embed, self.dropout_embed, training=self.training)

                embed = self.embed_layer2(embed)
                if self.dropout_embed:
                    embed = F.dropout(embed, self.dropout_embed, training=self.training)

                embed = torch.mean(embed, dim=0)
                embed = self.embed_layer3(embed)
                if self.dropout_embed:
                    embed = F.dropout(embed, self.dropout_embed, training=self.training)
                scen_embed.append(embed.unsqueeze(0))
            scen_embed = torch.stack(scen_embed)
        return scen_embed
