import unittest

from algorithm.nn_models.layers.seq_layers import *

BATCH = 8
EMBED_DIM = 16


class TestEpisodeMultiheadAttention(unittest.TestCase):
    def test_no_hidden_state(self):
        attn = EpisodeMultiheadAttention(EMBED_DIM, 2, num_layers=2)
        attn(torch.rand((BATCH, 10, EMBED_DIM)),
             query_length=2)

    def test_no_pre_hidden_state(self):
        attn = EpisodeMultiheadAttention(EMBED_DIM, 2, num_layers=2)
        attn(torch.rand((BATCH, 10, EMBED_DIM)),
             query_length=2,
             hidden_state=torch.rand((BATCH, 11, EMBED_DIM)),
             is_prev_hidden_state=False,
             key_padding_mask=torch.randint(0, 2, (BATCH, 10), dtype=torch.bool))

    def test_pre_hidden_state(self):
        attn = EpisodeMultiheadAttention(EMBED_DIM, 2, num_layers=2)
        attn(torch.rand((BATCH, 10, EMBED_DIM)),
             query_length=2,
             hidden_state=torch.rand((BATCH, 2, EMBED_DIM)),
             is_prev_hidden_state=True,
             key_padding_mask=torch.randint(0, 2, (BATCH, 10), dtype=torch.bool))
