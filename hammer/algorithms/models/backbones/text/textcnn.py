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
class TextCNN(torch.nn.Module):

    def __init__(self, config: Config, logger: Logger):
        """initialize TextCNN.

        Args:
            config (Config): configuration for backbone.
            logger (Logger): instance of logger.
        """
        super(TextCNN, self).__init__()

        # create embeddings from pretrained embeddings
        if config.dataset.pretrained_embeddings:
            self.embeddings = torch.nn.Embedding.from_pretrained(
                embeddings = np.load(config.dataset.pretrained_embeddings).astype('float32'),
                freeze = config.model.fixed_embeddings
            )

        # initialize embeddings randomly
        else:
            self.embeddings = torch.nn.Embedding(
                num_embeddings = config.dataset.vocab_size,
                embedding_dim = config.model.embedding_dim,
                padding_idx = config.dataset.vocab_size - 1
            )

        # convolutional layer
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(
                in_channels = config.model.embedding_dim,
                out_channels = num_filter,
                kernel_size = filter_size,
                padding = 'same'
            ) for num_filter, filter_size in zip(config.model.num_filters, config.model.filter_sizes)
        ])

        # dropout layer
        self.dropout = torch.nn.Dropout(config.model.dropout)

    def forward(self, inputs: dict) -> dict:
        """forward function of backbone.

        Args:
            inputs (dict): inputs of backbone.

        Returns:
            dict: outputs of backbone.
        """
        outputs = {}

        # lookup embeddings
        input_embeddings = self.embeddings(inputs['tokens'])

        # mask input embeddings according to sequence length
        batch_size = input_embeddings.size(0)
        max_length = self.config.dataset.max_num_tokens
        indexes = torch.arange(0, max_length).float().unsqueeze(0)
        sequence_mask = indexes.expand(batch_size, max_length).lt(inputs['length'].unsqueeze(1))
        masked_embeddings = input_embeddings * sequence_mask.unsqueeze(-1)

        # convolutional operation on input embeddings
        transposed_embeddings = torch.transpose(masked_embeddings, 1, 2)
        sequence_hiddens = [torch.nn.functional.relu(conv(transposed_embeddings)) for conv in self.convs]
        sequence_hidden = torch.cat(sequence_hiddens, 1)
        outputs['sequence_outputs'] = torch.transpose(sequence_hidden, 1, 2)

        # pooled outputs
        pooled_outputs = torch.nn.functional.max_pool1d(sequence_hidden, sequence_hidden.size(2)).squeeze(2)
        outputs['pooled_outputs'] = self.dropout(pooled_outputs)

        return outputs
