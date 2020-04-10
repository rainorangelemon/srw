import torch


def clipped_sum_error(y_true, y_pred):
    """
    Gradient Cipping as in DQN Paper
    """
    errs = y_pred - y_true
    quad = torch.clamp(abs(errs), max=1)
    lin = abs(errs) - quad
    return torch.sum(0.5 * quad ** 2 + lin)


def slice_tensor_tensor(tensor, tensor_slice):
    """
        Theano and tensorflow differ in the method of extracting the value of the actions taken
        arg1: the tensor to be slice i.e Q(s)
        arg2: the indices to slice by ie a
    """
    return select_at_indexes(tensor_slice, tensor)


def select_at_indexes(indexes, tensor):
    """Returns the contents of ``tensor`` at the multi-dimensional integer
    array ``indexes``. Leading dimensions of ``tensor`` must match the
    dimensions of ``indexes``.
    """
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    s_flat = t_flat[torch.arange(num), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1:])
