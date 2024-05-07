#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
from logging import Logger

# Third-party libraries
import torch
import numpy as np

# User define module
from hammer.utils.config import Config

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class GCN(torch.nn.Module):

    def __init__(self, input_dim: int, config: Config, logger: Logger):
        """initialize TextCNN.

        Args:
            config (Config): configuration for backbone.
            logger (Logger): instance of logger.
        """
        self.weights = []
        # weights for grach convolutional layers
        for _ in range(config.model.gcn.num_layers):
            weight = torch.nn.Parameter(torch.randn(input_dim, config.model.gcn.hidden_size))
            torch.nn.init.xavier_uniform_(weight)
            self.weights.append(weight)
        torch.nn.init.xavier_uniform_(self.weight)
        # layer normalization if enabled
        if config.model.gcn.apply_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(normalized_shape=[config.model.gcn.hidden_size])
        else:
            self.layer_norm = None
        # dropout layer
        self.dropout = torch.nn.Dropout(config.model.gcn.dropout)

    def _norm_adj_matrix(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """normalize adjacency matrix.

        Args:
            adj_matrix (torch.Tensor): adjacency matrix.

        Returns:
            torch.Tensor: normalized adjacency matrix.
        """
        # calculate the sum of each row
        row_sum = torch.sum(adj_matrix, dim=1)
        # inverse the sum of each row
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        # construct the diagonal matrix
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        # normalize the adjacency matrix
        adj_matrix = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj_matrix), d_mat_inv_sqrt)

        return adj_matrix
    
    def _attention_pooling(self, nodes: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            nodes (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        query = key = value = nodes
        # calculate the attention score
        logits = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(query.size(-1))
        # normalize the attention score
        attention_score = torch.nn.functional.softmax(logits, dim=-1)
        # calculate the weighted sum of the input nodes
        output = torch.matmul(attention_score, value)

        return output

    def forward(self, inputs: dict) -> dict:
        """forward function of backbone.

        Args:
            inputs (dict): inputs of backbone.

        Returns:
            dict: outputs of backbone.
        """
        outputs = {}
        
        hidden = inputs['input_embeddings']
        # normalize adjacency matrix
        adj_matrix = self._norm_adj_matrix(inputs['adj_matrix'])
        for weight in self.weights:
            # mask input embeddings according to sequence length
            hidden = torch.matmul( weight)
            # convolutional operation on input embeddings
            hidden = torch.matmul(adj_matrix, hidden)
        #
        outputs['output'] = self.dropout(hidden)

        return outputs
