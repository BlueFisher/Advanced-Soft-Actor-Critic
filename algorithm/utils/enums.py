from enum import Enum


class SEQ_ENCODER(Enum):
    RNN = 1
    ATTN = 2


class SIAMESE(Enum):
    ATC = 1
    BYOL = 2


class CURIOSITY(Enum):
    FORWARD = 1
    INVERSE = 2


def convert_config_to_enum(config):
    if 'seq_encoder' in config and config['seq_encoder'] is not None:
        config['seq_encoder'] = SEQ_ENCODER[config['seq_encoder']]

    if 'siamese' in config and config['siamese'] is not None:
        config['siamese'] = SIAMESE[config['siamese']]

    if 'curiosity' in config and config['curiosity'] is not None:
        config['curiosity'] = CURIOSITY[config['curiosity']]


def convert_config_to_string(config):
    if 'seq_encoder' in config and config['seq_encoder'] is not None:
        config['seq_encoder'] = config['seq_encoder'].name

    if 'siamese' in config and config['siamese'] is not None:
        config['siamese'] = config['siamese'].name

    if 'curiosity' in config and config['curiosity'] is not None:
        config['curiosity'] = config['curiosity'].name
