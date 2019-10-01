This project is the implementation of [Soft Actor-Critic algorithm](https://arxiv.org/abs/1812.05905) algorithm and [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933) version of Soft Actor-Critic (currently disabled). The environment is based on [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).

The whole project is now migrated to TensorFlow 2.0

Besides of standard properties of SAC, we added some improvements,

1. Prioritized Experience Replay
2. N-step SAC
3. N-step Importance Sampling
4. Recurrent Experience Replay