from copy import deepcopy
import numpy as np

from lib.utils import ExperienceReplay
from lib.torch_utils import slice_tensor_tensor, clipped_sum_error
from dqn_secure.model import AtariDqnModel
import torch

import logging

logger = logging.getLogger(__name__)


# use RMSProp


class AI(object):
    def __init__(self, state_shape, nb_actions, action_dim, reward_dim, no_network=False, history_len=1, gamma=.99,
                 learning_rate=0.00025, minibatch_size=32, update_freq=50, learning_frequency=1, ddqn=False,
                 normalize=1., replay_buffer=None, bootstrap_corr=(), rng=None, device="cpu"):
        self.rng = rng
        self.history_len = history_len
        self.state_shape = state_shape
        self.nb_actions = nb_actions
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.update_freq = update_freq
        self.update_counter = 0
        self.normalize = normalize
        self.learning_frequency = learning_frequency
        self.transitions = replay_buffer
        self.rewarding_transitions = []
        self.ddqn = ddqn
        self.bootstrap_corr = bootstrap_corr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not no_network:
            self.network = self._build_network()
            self.target_network = self._build_network()
            self.weight_transfer(from_model=self.network, to_model=self.target_network)
        else:
            self.network = None
            self.target_network = None
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=.95, eps=1e-7)
        self._compile_learning()
        logger.warning('Compiled Model and Learning.')

    def _build_network(self):
        """ returns the keras nn compiled model """
        return AtariDqnModel(self.state_shape, self.nb_actions).to(self.device)

    def _compile_learning(self):

        self.predict_network = lambda x: self.network(x)
        self.predict_target = lambda x: self.target_network(x)
        self.update_weights = lambda x: self.weight_transfer(self.network, self.target_network)

    def _train_on_batch(self, s, a, r, s2, t):
        a = a.long()
        # Q(s, a)
        q = self.network(s)
        preds = slice_tensor_tensor(q, a)

        # r + (1 - t) * gamma * max_a(Q'(s'))
        q2 = self.target_network(s2)
        if self.ddqn:
            q2_net = self.network(s2).detach()
            a_max = torch.argmax(q2_net, dim=1)
            q2_max = slice_tensor_tensor(q2, a_max)
        else:
            q2_max = torch.max(q2, dim=1)

        # over-estimation correction
        if len(self.bootstrap_corr) > 0:
            q2_max -= (q2_max - np.float32(self.bootstrap_corr[1])) * (q2_max > self.bootstrap_corr[1])
            q2_max -= (q2_max - np.float32(self.bootstrap_corr[0])) * (q2_max < self.bootstrap_corr[0])

        targets = r + (1 - t.float()) * self.gamma * q2_max

        # Loss and Updates
        loss = clipped_sum_error(y_true=targets, y_pred=preds)

        return loss

    def buffer_to(self, buffer_):
        if isinstance(buffer_, torch.Tensor):
            return buffer_.to(self.device)

    def get_q(self, states, target):
        states = torch.tensor(self._reshape(states))
        states = self.buffer_to(states)
        if not target:
            return self.predict_network(states)
        else:
            return self.predict_target(states)

    def get_max_action(self, states, target):
        states = torch.tensor(self._reshape(states))
        states = self.buffer_to(states)
        if not target:
            return np.argmax(self.predict_network(states)[0], axis=1)
        else:
            return np.argmax(self.predict_target(states)[0], axis=1)

    def get_safe_actions(self, states, target, q_threshold):
        q = self.get_q(states, target)[0]
        q = q > q_threshold
        return np.where(q == True)[0]  # could be empty

    def get_secure_uniform_action(self, s):
        # Uniform and secure (presumption of innocence)
        q = self.get_q(s, target=False).to("cpu").detach().numpy().astype(np.float64)
        q[q < -1] = -1.0
        q[q > 0] = 0.0
        if all(abs(q + 1) < 0.01):  # if all values are -1
            return self.rng.randint(0, self.nb_actions)
        else:
            eta = (1.0 + q) / (self.nb_actions + np.sum(q))
            selector = self.rng.multinomial(1, eta)
            return int(np.where(selector == 1)[0])

    def get_safe_max_actions(self, states, target, q_threshold):
        q = self.get_q(states, target).to("cpu").detach().numpy().astype(np.float64)
        safe_q = q[q > q_threshold]
        if len(safe_q) > 0:
            actions = np.where(q == np.max(safe_q))[0]
        else:
            actions = []
        return actions

    def train_on_batch(self, s, a, r, s2, t):
        if self.action_dim == 1:
            a = a.flatten()
        return self._train_on_batch(s, a, r, s2, t)

    def target_network_update(self):
        # updates the target network to the main network weights
        self.update_weights([])

    def learn(self, goal_prob):
        """
        Learning from one minibatch
        NOTE: with probability `goal_prob`, it puts a rewarding transition in the minibatch
        """
        assert self.minibatch_size <= self.transitions.size, 'not enough data in the pool'
        # sampling one minibatch
        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)
        nb_rew_trans = len(self.rewarding_transitions)
        if nb_rew_trans >= 1 and bool(self.rng.binomial(1, goal_prob)):
            randint = self.rng.randint(0, nb_rew_trans)
            s_1, a_1, r_1, s2_1, term_1 = self.rewarding_transitions[randint]
            s[-1] = s_1
            a[-1] = a_1
            r[-1] = r_1
            s2[-1] = s2_1
            term[-1] = term_1
        s, a, r, s2, term = self.to_device((s, a, r, s2, term))
        self.optimizer.zero_grad()
        # TODO: make s, a, r, s2, term as tensor
        objective = self.train_on_batch(s, a, r, s2, term)
        objective.backward()
        self.optimizer.step()
        # updating target network
        if self.update_counter == self.update_freq:
            self.target_network_update()
            self.update_counter = 0
        else:
            self.update_counter += 1
        return objective

    def to_device(self, buffer_):
        """Call method ``method_name(*args, **kwargs)`` on all contents of
        ``buffer_``, and return the results. ``buffer_`` can be an arbitrary
        structure of tuples, namedtuples, namedarraytuples, NamedTuples, and
        NamedArrayTuples, and a new, matching structure will be returned.
        ``None`` fields remain ``None``.
        """
        if buffer_ is None:
            return
        if isinstance(buffer_, np.ndarray):
            return torch.from_numpy(buffer_).to(self.device)
        if isinstance(buffer_, torch.Tensor):
            return buffer_.to(self.device)
        contents = tuple(self.to_device(b) for b in buffer_)
        if type(buffer_) is tuple:
            return contents
        return buffer_._make(contents)

    def dump_network(self, weights_file_path='q_network_weights.h5', overwrite=True):
        self.network.save_weights(weights_file_path, overwrite=overwrite)

    def load_weights(self, weights_file_path='q_network_weights.h5', target=False):
        self.network.load_weights(weights_file_path)
        if target:
            self.target_network_update()

    def _reshape(self, states):
        if len(self.state_shape) + 1 == states.ndim:
            shape = [1] + list(states.shape)
            states = states.reshape(shape)
        return states

    @staticmethod
    def weight_transfer(from_model, to_model):
        """Update the state dict of ``model`` using the input ``state_dict``, which
        must match format.  ``tau==1`` applies hard update, copying the values, ``0<tau<1``
        applies soft update: ``tau * new + (1 - tau) * old``.
        """
        state_dict = from_model.state_dict()
        to_model.load_state_dict(state_dict)
