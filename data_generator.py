import numpy as np
import torch
from utils import torch_to_numpy
from collections import deque

class DataGenerator:
    """
    A data generator used to collect trajectories for on-policy RL with GAE
    References:
        https://github.com/Khrylx/PyTorch-RL
        https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
        https://github.com/ikostrikov/pytorch-trpo
    """
    def __init__(self, aug_obs_dim, act_dim, batch_size, max_eps_len):

        # Hyperparameters
        self.aug_obs_dim = aug_obs_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.max_eps_len = max_eps_len

        # Batch buffer
        self.obs_buf = np.zeros((batch_size, aug_obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((batch_size, act_dim),  dtype=np.float32)
        self.vtarg_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.adv_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.cvtarg_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.cadv_buf = np.zeros((batch_size, 1), dtype=np.float32)

        # Episode buffer
        self.obs_eps = np.zeros((max_eps_len, aug_obs_dim),  dtype=np.float32)
        self.next_obs_eps = np.zeros((max_eps_len, aug_obs_dim),  dtype=np.float32)
        self.act_eps = np.zeros((max_eps_len, act_dim),  dtype=np.float32)
        self.rew_eps = np.zeros((max_eps_len, 1),  dtype=np.float32)
        self.cost_eps = np.zeros((max_eps_len, 1), dtype=np.float32)
        self.eps_len = 0
        self.not_terminal = 1


        # Pointer
        self.ptr = 0

    # New helper method:
    def get_augmented_state(self, state, action_buffer):
        """Convert raw state to augmented state with action history"""
        if len(action_buffer) != 0:
            action_history = np.concatenate(list(action_buffer), axis=0)
            aug_state = np.concatenate([state, action_history])
            return aug_state
        else:
            return state

    def run_traj(self, env, policy, value_net, cvalue_net, delay_steps, running_stat,
                 score_queue, cscore_queue, gamma, c_gamma, gae_lam, c_gae_lam,
                 dtype, device):

        batch_idx = 0

        cost_ret_hist = []

        avg_eps_len = 0
        num_eps = 0

        while batch_idx < self.batch_size:
            obs, _ = env.reset()
            if running_stat is not None:
                obs = running_stat.normalize(obs)

            # initialize action buffer for delay
            action_buffer = deque(maxlen=delay_steps)
            for _ in range(delay_steps):
                action_buffer.append(np.zeros(self.act_dim))

            # get initial augmented state
            aug_obs = self.get_augmented_state(obs, action_buffer)

            ret_eps = 0
            cost_ret_eps = 0

            for t in range(self.max_eps_len):
                act = policy.get_act(torch.Tensor(aug_obs).to(dtype).to(device))
                act = torch_to_numpy(act).squeeze()
                next_obs, rew, cost, terminated, truncated, _ = env.step(act)
                done = np.logical_or(terminated, truncated)

                ret_eps += rew
                cost_ret_eps += (c_gamma ** t) * cost

                # Update action buffer and get next augmented state
                action_buffer.append(act)

                if running_stat is not None:
                    next_obs = running_stat.normalize(next_obs)
                next_aug_obs = self.get_augmented_state(next_obs, action_buffer)

                # Store in episode buffer
                self.obs_eps[t] = aug_obs
                self.act_eps[t] = act
                self.next_obs_eps[t] = next_aug_obs
                self.rew_eps[t] = rew
                self.cost_eps[t] = cost

                aug_obs = next_aug_obs

                self.eps_len += 1
                batch_idx += 1

                # Store return for score if only episode is terminal
                if done or t == self.max_eps_len - 1:
                    if done:
                        self.not_terminal = 0
                    score_queue.append(ret_eps)
                    cscore_queue.append(cost_ret_eps)
                    cost_ret_hist.append(cost_ret_eps)

                    num_eps += 1
                    avg_eps_len += (self.eps_len - avg_eps_len) / num_eps

                if done or batch_idx == self.batch_size:
                    break

            # Store episode buffer
            self.obs_eps, self.next_obs_eps = self.obs_eps[:self.eps_len], self.next_obs_eps[:self.eps_len]
            self.act_eps, self.rew_eps = self.act_eps[:self.eps_len], self.rew_eps[:self.eps_len]
            self.cost_eps = self.cost_eps[:self.eps_len]


            # Calculate advantage
            adv_eps, vtarg_eps = self.get_advantage(value_net, gamma, gae_lam, dtype, device, mode='reward')
            cadv_eps, cvtarg_eps = self.get_advantage(cvalue_net, c_gamma, c_gae_lam, dtype, device, mode='cost')


            # Update batch buffer
            start_idx, end_idx = self.ptr, self.ptr + self.eps_len
            self.obs_buf[start_idx: end_idx], self.act_buf[start_idx: end_idx] = self.obs_eps, self.act_eps
            self.vtarg_buf[start_idx: end_idx], self.adv_buf[start_idx: end_idx] = vtarg_eps, adv_eps
            self.cvtarg_buf[start_idx: end_idx], self.cadv_buf[start_idx: end_idx] = cvtarg_eps, cadv_eps


            # Update pointer
            self.ptr = end_idx

            # Reset episode buffer and update pointer
            self.obs_eps = np.zeros((self.max_eps_len, self.aug_obs_dim), dtype=np.float32)
            self.next_obs_eps = np.zeros((self.max_eps_len, self.aug_obs_dim), dtype=np.float32)
            self.act_eps = np.zeros((self.max_eps_len, self.act_dim), dtype=np.float32)
            self.rew_eps = np.zeros((self.max_eps_len, 1), dtype=np.float32)
            self.cost_eps = np.zeros((self.max_eps_len, 1), dtype=np.float32)
            self.eps_len = 0
            self.not_terminal = 1

        avg_cost = np.mean(cost_ret_hist)
        std_cost = np.std(cost_ret_hist)


        # Normalize advantage functions
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-6)
        self.cadv_buf = (self.cadv_buf - self.cadv_buf.mean()) / (self.cadv_buf.std() + 1e-6)


        return {'states':self.obs_buf, 'actions':self.act_buf,
                'v_targets': self.vtarg_buf,'advantages': self.adv_buf,
                'cv_targets': self.cvtarg_buf, 'c_advantages': self.cadv_buf,
                'avg_cost': avg_cost, 'std_cost': std_cost, 'avg_eps_len': avg_eps_len}


    def get_advantage(self, value_net, gamma, gae_lam, dtype, device, mode='reward'):
        gae_delta = np.zeros((self.eps_len, 1))
        adv_eps =  np.zeros((self.eps_len, 1))
        # Check if terminal state, if terminal V(S_T) = 0, else V(S_T)
        status = np.ones((self.eps_len, 1))
        status[-1] = self.not_terminal
        prev_adv = 0

        for t in reversed(range(self.eps_len)):
            # Get value for current and next state
            obs_tensor = torch.Tensor(self.obs_eps[t]).to(dtype).to(device)
            next_obs_tensor = torch.Tensor(self.next_obs_eps[t]).to(dtype).to(device)
            current_val, next_val = torch_to_numpy(value_net(obs_tensor), value_net(next_obs_tensor))

            # Calculate delta and advantage
            if mode == 'reward':
                gae_delta[t] = self.rew_eps[t] + gamma * next_val * status[t] - current_val
            elif mode =='cost':
                gae_delta[t] = self.cost_eps[t] + gamma * next_val * status[t] - current_val
            adv_eps[t] = gae_delta[t] + gamma * gae_lam * prev_adv

            # Update previous advantage
            prev_adv = adv_eps[t]

        # Get target for value function
        obs_eps_tensor = torch.Tensor(self.obs_eps).to(dtype).to(device)
        vtarg_eps = torch_to_numpy(value_net(obs_eps_tensor)) + adv_eps



        return adv_eps, vtarg_eps
