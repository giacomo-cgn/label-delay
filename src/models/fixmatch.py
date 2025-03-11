import torch
from torch import nn
import torch.nn.functional as F

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class FixMatch(nn.Module):
    def __init__(self,
             num_classes: int,
             encoder: nn.Module,
             pseudo_label_temp: float = 1.0,
             pseudo_label_thresh: float = 0.95,
             omega_pseudo: float = 1.0,
             ):

        super(FixMatch, self).__init__()

        self.encoder = encoder
        self.features_dim = self.encoder.features_dim
        self.pseudo_label_temp = pseudo_label_temp
        self.pseudo_label_thresh = pseudo_label_thresh
        self.omega_pseudo = omega_pseudo


        self.clf = nn.Linear(self.features_dim, num_classes)


    def forward(self, s_x1, s_x2, s_y, u_curr_x_weak, u_curr_x_strong, u_buff_x_weak, u_buff_strong):

        s_batch_size = s_x1.size(0)
        u_batch_size = u_curr_x_weak.size(0)

        unsup_sup_ratio = u_batch_size // s_batch_size
        

        inputs = interleave(
            torch.cat((s_x1, u_curr_x_weak, u_curr_x_strong)), 2*unsup_sup_ratio+1)
        logits = self.encoder(inputs)
        logits = de_interleave(logits, 2*unsup_sup_ratio+1)
        logits_x = logits[:s_batch_size]
        logits_u_w, logits_u_s = logits[s_batch_size:].chunk(2)
        del logits

        Lx = F.cross_entropy(logits_x, s_y, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach()/self.pseudo_label_temp, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.pseudo_label_thresh).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u,
                                reduction='none') * mask).mean()

        loss = Lx + self.omega_pseudo * Lu

        return loss, Lx, Lu, 0




    def forward_eval(self, x):
        return self.clf(self.encoder(x))


