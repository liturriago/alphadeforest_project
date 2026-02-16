import torch
import torch.nn.functional as F

def get_anomaly_scores(model, x_sequence, lambda_rec=1.0, lambda_pred=0.5):
    model.eval()
    device = next(model.parameters()).device

    x_sequence = x_sequence.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x_sequence)

    x_rec = outputs["reconstructions"].squeeze(0)
    z_f = outputs["z_f"].squeeze(0)
    z_pred = outputs["z_pred"].squeeze(0)

    scores = []
    for t in range(1, x_rec.shape[0]):
        rec_err = F.mse_loss(x_rec[t], x_sequence.squeeze(0)[t])
        pred_err = F.mse_loss(z_pred[t - 1], z_f[t])

        score = lambda_rec * rec_err + lambda_pred * pred_err
        scores.append(score.item())

    return scores
