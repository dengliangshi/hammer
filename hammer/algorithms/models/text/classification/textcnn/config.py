#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2023
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library


# Third-party libraries


# User define module
from hammer.utils.attr_dict import AttrDict
from hammer.utils.config import Config as BaseConfig

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Config(BaseConfig):

    def __init__(self, config_file: str):
        """initialize the configuration using user-defined parameters.

        Args:
            config_file (str): path of configuration file.
        """
        self.model_name = 'TextCNN'
        self.dataset.voab_size = 2

        self.textcnn = AttrDict()
        # if word embeddings are trainable
        self.textcnn.fixed_embeddings = False
        # embedding dimension
        self.textcnn.embedding_dim = 128
        # number of filters
        self.textcnn.num_filters = [32, 64, 128, 256]
        # size of filters
        self.textcnn.filter_sizes = [2, 3, 4, 5]
        # keep probabilty of dropout
        self.textcnn.dropout = 0.5

        self.train = AttrDict()
        #
        self.train.learning_rate = 1e-3

        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)

        super(Config, self).__init__(config_file)