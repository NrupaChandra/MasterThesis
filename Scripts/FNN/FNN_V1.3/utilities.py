import torch

def compute_integration(nodes_x, nodes_y, weights, test_fn):

    integral = torch.sum(weights * test_fn(nodes_x, nodes_y), 1)
    return integral

