import torch
from torch import nn
import torch.nn.functional as F


class HybridSSLSemi(nn.Module):
    def __init__(self,
             num_classes: int,
             encoder: nn.Module,
             proj_hidden_dim: int = 2048,
             proj_output_dim: int = 2048,
             adapter_hidden_dim: int = 2048,
             lambda_barlow: float = 5e-3,
             pseudo_label_temp: float = 1.0,
             pseudo_label_thresh: float = 0.95,
             omega_pseudo: float = 1.0,
             omega_ssl: float = 1.0
             ):

        super(HybridSSLSemi, self).__init__()

        self.encoder = encoder
        self.features_dim = self.encoder.features_dim
        self.lambda_barlow = lambda_barlow
        self.pseudo_label_temp = pseudo_label_temp
        self.pseudo_label_thresh = pseudo_label_thresh
        self.omega_pseudo = omega_pseudo
        self.omega_ssl = omega_ssl

        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim, bias=False),
        )

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(proj_output_dim, affine=False)

        self.adapter = nn.Sequential(
            nn.Linear(self.features_dim, adapter_hidden_dim, bias=False),
            nn.BatchNorm1d(adapter_hidden_dim),
            nn.ReLU(),
            nn.Linear(adapter_hidden_dim, self.features_dim, bias=False),
        )

        self.clf = nn.Linear(self.features_dim, num_classes)

        def barlow_criterion(z1, z2):
            # empirical cross-correlation matrix
            c = self.bn(z1).T @ self.bn(z2)

            # sum the cross-correlation matrix between all gpus
            batch_size = z1.shape[0]
            c.div_(batch_size)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            return on_diag + self.lambda_barlow * off_diag
        self.barlow_criterion = barlow_criterion

    def forward(self, s_x1, s_x2, s_y, u_curr_x_weak, u_curr_x_strong, u_buff_x_weak, u_buff_strong):
        all_x1, all_x2 = None, None

        # Check supervised data
        if not s_x1 == None and not s_x2 == None:
            assert s_x1.size(0) == s_x2.size(0)
            len_sup = s_x1.size(0)
            all_x1 = s_x1
            all_x2 = s_x2
        else:
            len_sup = 0

        # Check unsupervised data from buffer
        if not u_buff_x_weak == None and not u_buff_strong == None:
            assert u_buff_x_weak.size(0) == u_buff_strong.size(0)
            len_unsup_buff = u_buff_x_weak.size(0)
            if all_x1 == None:
                all_x1 = u_buff_x_weak
                all_x2 = u_buff_strong
            else:
                all_x1 = torch.cat((all_x1, u_buff_x_weak))
                all_x2 = torch.cat((all_x2, u_buff_strong))
        else:
            len_unsup_buff = 0


        #  Check unnsupervised data from current experience
        if not u_curr_x_weak == None and not u_curr_x_strong == None:
            assert u_curr_x_weak.size(0) == u_curr_x_strong.size(0)
            len_unsup_curr = u_curr_x_weak.size(0)
            if all_x1 == None:
                all_x1 = u_curr_x_weak
                all_x2 = u_curr_x_strong
            else:
                all_x1 = torch.cat((all_x1, u_curr_x_weak))
                all_x2 = torch.cat((all_x2, u_curr_x_strong))
        else:
            len_unsup_curr = 0

        # Minibatch is composed by [s_x, u_buff_x, u_curr_x]
            
        # Extract encoder features
        e1, e2 = self.encoder(all_x1), self.encoder(all_x2)
        # Pass samples through projector
        z1, z2 = self.projector(e1), self.projector(e2)
        z1, z2 = self.bn(z1), self.bn(z2)
        # Compute Barlow loss
        loss_barlow = self.barlow_criterion(z1, z2)

        # Only s_x and u_buff_x are passed through adapter and classifier for pseudo-labeling and supervised loss
        if len_sup > 0 and len_unsup_buff > 0:
            pseudo_e1, pseudo_e2 = e1[:len_sup+len_unsup_buff], e2[:len_sup+len_unsup_buff]
            pseudo_a1, pseudo_a2 = self.adapter(pseudo_e1), self.adapter(pseudo_e2)
            l1, l2 = self.clf(pseudo_a1), self.clf(pseudo_a2)

            loss_supervised = (F.cross_entropy(l1[:len_sup], s_y) + F.cross_entropy(l2[:len_sup], s_y)).mean()

            # Compute pseudolabeling loss on unsupervised buffer logits
            l_u_buff_weak = l1[len_sup:len_sup+len_unsup_buff]
            l_u_buff_strong = l2[len_sup:len_sup+len_unsup_buff]
            
            pseudo_label = torch.softmax(l_u_buff_weak.detach()/self.pseudo_label_temp, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.pseudo_label_thresh).float()
            loss_pseudo = (F.cross_entropy(l_u_buff_strong, targets_u,
                                    reduction='none') * mask).mean()
        else:
            loss_pseudo = 0.0
            loss_supervised = 0.0

        
        loss = loss_supervised + self.omega_pseudo * loss_pseudo + self.omega_ssl * loss_barlow

        return loss, loss_supervised, loss_pseudo, loss_barlow
    
    def forward_eval(self, x):
        return self.clf(self.adapter(self.encoder(x)))





def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
