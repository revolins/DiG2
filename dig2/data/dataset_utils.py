import pickle

import torch


def save_pickle(array, path):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class DistributionNodes:
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)
        # Max representable node count (prob[i] = P(n_nodes == i), so this is
        # prob's last valid index). Several call sites -- e.g. sample()'s
        # max_nodes_per_batch budgeting in graph_discrete_interpolative_model.py -- read
        # this via getattr(node_dist, "max_n", ...) and silently fall back to a
        # wrong default if it's missing, so keep it defined even though nothing
        # in this class itself uses it.
        self.max_n = int(self.prob.numel() - 1)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

