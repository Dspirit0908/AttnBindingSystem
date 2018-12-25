# coding: utf-8

import torch
from utils import sequence_mask
from models.baseline import Baseline
from torch.distributions import Categorical


class Policy:
    def __init__(self, args):
        self.args = args

    def select_action_step(self, probs_step):
        m = Categorical(probs_step)
        action = m.sample()
        return action, m.log_prob(action)

    def select_action(self, probs, lengths, sql_labels):
        batch_size = probs.size(0)
        # notice the max_len
        max_len = probs.size(1)
        actions, log_probs = torch.LongTensor(batch_size, max_len).to(device=self.args.device), torch.FloatTensor(batch_size, max_len).to(device=self.args.device)
        probs = probs.transpose(0, 1).contiguous()
        for ti in range(max_len):
            actions[:, ti], log_probs[:, ti] = self.select_action_step(probs[ti])
        # mask
        mask = sequence_mask(lengths, max_len).to(device=self.args.device)
        actions.data.masked_fill_(1 - mask, -100)
        log_probs.data.masked_fill_(1 - mask, 0.0)
        # compute rewards
        rewards = self.compute_rewards(actions, sql_labels)
        return torch.sum(log_probs, dim=1), rewards

    def compute_rewards_step(self, action, sel_col):
        action_cols = [tag - 2 * self.args.tokenize_max_len for tag in action.data.cpu().numpy() if tag >= 2 * self.args.tokenize_max_len]
        reward = 1.0 if sel_col.data.cpu().numpy() in action_cols else -1.0
        return reward

    def compute_rewards(self, actions, sql_labels):
        batch_size = actions.size(0)
        rewards = torch.FloatTensor(batch_size).to(device=self.args.device)
        for bi in range(batch_size):
            pred, true = actions[bi], sql_labels[bi]
            rewards[bi] = self.compute_rewards_step(pred, true)
        return rewards


if __name__ == '__main__':
    probs = torch.Tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
    print(torch.sum(probs, dim=1))
