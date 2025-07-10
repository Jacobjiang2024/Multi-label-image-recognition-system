import torch
def asymmetric_loss_with_logits(pred, target, gamma_neg=4, clip=0.05):
    """
    Asymmetric loss for multi-label classification.
    pred: logits
    target: binary labels
    """
    pred_sigmoid = torch.sigmoid(pred)
    target = target.type_as(pred)

    # Clip predicted probs to avoid log(0)
    if clip is not None and clip > 0:
        pred_sigmoid = torch.clamp(pred_sigmoid, min=clip, max=1 - clip)

    pos_loss = target * torch.log(pred_sigmoid)
    neg_loss = (1 - target) * ((1 - pred_sigmoid) ** gamma_neg) * torch.log(1 - pred_sigmoid)

    loss = - (pos_loss + neg_loss)
    return loss.mean()