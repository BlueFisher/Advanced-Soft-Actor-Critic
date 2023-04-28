from algorithm.oc.oc_agent import *

obs_shapes = [(4,)]
c_action_size = 2
attn_state_size = (4,)
low_rnn_state_size = (6,)
agent = OC_Agent(0, obs_shapes, [], c_action_size, attn_state_size, low_rnn_state_size)


def print_episode_trans(episode_trans):
    for e in episode_trans:
        if isinstance(e, list):
            print([o.shape for o in e])
        elif e is not None:
            print(e.shape)
        else:
            print('None')


attn_state = np.random.randn(*attn_state_size)
low_rnn_state = np.random.randn(*low_rnn_state_size)
for i in range(10):
    a = agent.add_transition([np.random.randn(*s) for s in obs_shapes],
                             -1,
                             np.random.randn(c_action_size), 0., False, False,
                             [np.random.randn(*s) for s in obs_shapes],
                             np.random.rand(c_action_size),
                             is_padding=True,
                             seq_hidden_state=attn_state,
                             low_seq_hidden_state=low_rnn_state)

for i in range(2):
    a = agent.add_transition([np.random.randn(*s) for s in obs_shapes],
                             0,
                             np.random.randn(c_action_size), 0., False, False,
                             [np.random.randn(*s) for s in obs_shapes],
                             np.random.rand(c_action_size),
                             is_padding=False,
                             seq_hidden_state=attn_state,
                             low_seq_hidden_state=low_rnn_state)

for i in range(3):
    a = agent.add_transition([np.random.randn(*s) for s in obs_shapes],
                             1,
                             np.random.randn(c_action_size), 0., False, False,
                             [np.random.randn(*s) for s in obs_shapes],
                             np.random.rand(c_action_size),
                             is_padding=False,
                             seq_hidden_state=attn_state,
                             low_seq_hidden_state=low_rnn_state)

for i in range(1):
    a = agent.add_transition([np.random.randn(*s) for s in obs_shapes],
                             2,
                             np.random.randn(c_action_size), 0., False, False,
                             [np.random.randn(*s) for s in obs_shapes],
                             np.random.rand(c_action_size),
                             is_padding=False,
                             seq_hidden_state=attn_state,
                             low_seq_hidden_state=low_rnn_state)
