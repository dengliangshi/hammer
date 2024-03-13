#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2023
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
from logging import Logger

# Third-party libraries
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from torchmetrics.functional.classification import accuracy

# User define module
from hammer.utils.config import Config
from hammer.algorithms.models.modeling import Modeling
from hammer.algorithms.models.backbone import Backbone

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class TextCNN(Backbone):

    def __init__(self, config: Config, logger: Logger):
        """initialize TextCNN.

        Args:
            config (Config): configuration for backbone.
            logger (Logger): instance of logger.
        """
        super(TextCNN, self).__init__(config, logger)

        # create embeddings from pretrained embeddings
        if config.dataset.pretrained_embeddings:
            self.embeddings = torch.nn.Embedding.from_pretrained(
                embeddings = np.load(config.dataset.pretrained_embeddings).astype('float32'),
                freeze = config.textcnn.fixed_embeddings
            )

        # initialize embeddings randomly
        else:
            self.embeddings = torch.nn.Embedding(
                num_embeddings = config.dataset.vocab_size,
                embedding_dim = config.textcnn.embedding_dim,
                padding_idx = config.dataset.vocab_size - 1
            )

        # convolutional layer
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(
                in_channels = config.textcnn.embedding_dim,
                out_channels = num_filter,
                kernel_size = filter_size,
                padding = 'same'
            ) for num_filter, filter_size in zip(config.textcnn.num_filters, config.textcnn.filter_sizes)
        ])

        # dropout layer
        self.dropout = torch.nn.Dropout(config.textcnn.dropout)

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


class TextCNNForClassification(Modeling):

    def __init__(self, config: Config, logger: Logger):
        """initialize model.

        Args:
            config (Config): configuration for model.
            logger (Logger): instance of logger.
        """
        super(TextCNNForClassification, self).__init__(config, logger)

        self.textcnn = TextCNN(config, logger)

        self.fc = torch.nn.Linear(sum(config.textcnn.num_filters), config.dataset.num_classes)

    def forward(self, inputs: dict) -> dict:
        """forward function of model.

        Args:
            inputs (dict): inputs of model.

        Returns:
            dict: outputs of model.
        """
        outputs = {}

        hiddens = self.textcnn(inputs)

        outputs['logits'] = self.fc(hiddens['pooled_outputs'])

        return outputs

    def create_optimizer(self, learning_rate) -> Optimizer:
        """create optimizer for training model.

        Args:
            learning_rate (float): learning rate for trianing model.

        Returns:
            Optimizer: optimizer for training model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)

        return optimizer
    
    def create_lr_scheduler(self, learning_rate):
        """_summary_

        Args:
            learning_rate (_type_): _description_

        Returns:
            _type_: _description_
        """
        return learning_rate

    def loss(self, outputs: dict, targets: dict) -> float:
        """loss function of model.

        Args:
            outputs (dict): outputs of model.
            targets (dict): targets of given batch of samples.

        Returns:
            float: loss of given batch of samples.
        """
        return torch.nn.functional.cross_entropy(outputs['logits'], targets['class'])

    def metrics(self, outputs: dict, targets: dict) -> dict:
        """metrics of model.

        Args:
            outputs (dict): outputs of model.
            targets (dict): targets of given batch of samples.

        Returns:
            dict: metrics of given batch of samples.
        """
        predictions = torch.argmax(outputs['logits'], dim=1)

        acc = accuracy(
            preds = predictions, 
            target = targets['class'],
            task = 'multiclass',
            num_classes = self.config.dataset.num_classes
        )

        return {'acc': acc.item()}