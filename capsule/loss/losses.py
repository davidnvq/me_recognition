import torch


def me_loss(y_true, y_pred):
    """
    The loss function
    :param y_true: (tensor) of shape [N, num_classes]
    :param y_pred: (tensor) of shape [N, num_classes]
    :return:
    """
    L = y_true * torch.clamp(0.99 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.01, min=0.) ** 2

    # class_weights = torch.tensor([1.0, 10.97, 8.43]).cuda()
    # L = L * class_weights

    L_margin = L.sum(dim=1).mean()
    # print('loss:', L.sum(dim=0))
    return L_margin
