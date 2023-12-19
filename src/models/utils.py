from torch import nn
import torch.nn.functional as F


class KLLoss(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.T = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, output_batch, teacher_outputs):

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * self.kl_loss(output_batch, teacher_outputs)

        return loss
