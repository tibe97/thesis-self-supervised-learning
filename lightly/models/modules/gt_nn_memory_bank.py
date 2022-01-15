""" Nearest Neighbour Memory Bank Module with method for hard-negative sampling, using true labels. Done for validating the idea of false negative removal"""

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import time
import copy
import ipdb
import numpy as np
import random
from lightly.loss.memory_bank import MemoryBankModule


class GTNNMemoryBankModule(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation

    This class implements a nearest neighbour memory bank as described in the 
    NNCLR paper[0]. During the forward pass we return the nearest neighbour 
    from the memory bank.

    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548

    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.

    Examples:
        >>> model = NNCLR(backbone)
        >>> criterion = NTXentLoss(temperature=0.1)
        >>> 
        >>> nn_replacer = NNmemoryBankModule(size=2 ** 16)
        >>>
        >>> # forward pass
        >>> (z0, p0), (z1, p1) = model(x0, x1)
        >>> z0 = nn_replacer(z0.detach(), update=False)
        >>> z1 = nn_replacer(z1.detach(), update=True)
        >>>
        >>> loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))

    """
    def __init__(self, model, size: int = 2 ** 16, epsilon: float = 0.05, sinkhorn_iterations: int = 3, gpus: int = 0, 
                use_sinkhorn: bool = False, false_neg_remove: bool = False, soft_neg: bool = False):
        super(GTNNMemoryBankModule, self).__init__(size)
        self.model = model
        self.epsilon = epsilon #division coefficient for cluster assignemnt computation
        self.get_assignments = self.sinkhorn
        self.sinkhorn_iterations = sinkhorn_iterations
        self.gpus = gpus
        self.use_sinkhorn = use_sinkhorn
        self.false_neg_remove = false_neg_remove
        self.soft_neg = soft_neg
        

    def forward(self,
                output: torch.Tensor,
                y: torch.Tensor,
                num_nn: int,
                epoch: int = None,
                max_epochs: int = 400,
                update: bool = False):
        """Returns nearest neighbour of output tensor from memory bank which belongs to the same cluster
        as the reference

        Args:
            output: The torch tensor for which you want the nearest neighbour
            update: If `True` updated the memory bank by adding output to it

        """
        
        output, bank, y_bank = super(GTNNMemoryBankModule, self).forward(output, y, update=update)
        bank = bank.to(output.device).t()
        y_bank = y_bank.to(output.device).t()
        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

       
       

        # TODO: 1. pick random negatives; 2. batch + hard negatives as negatives
        negatives = []
        
       
        # efficient implementation of the loops with scatter_() function
        for i in range(output_normed.shape[0]): # for each positive sample
            p_class = y[i] # class of the positive being considered
            idx_bank_negatives = torch.where(y_bank!=p_class)[0]
            
            if not self.false_neg_remove:
                negatives.append(torch.index_select(bank_normed, dim=0, index=idx_bank_negatives))            
            else:
                # False Negatives removal from batch
                # remove false negatives from batch (i.e. positives) and replace them with samples
                # from memory bank
                mask_positives = torch.where(y==p_class)[0]
                num_false_negatives = mask_positives.shape[0]
                neg = output_normed
                if num_false_negatives > 0:
                    idx_positives = torch.Tensor(list(set(range(output.shape[0])) - set(mask_positives.tolist()))).to(torch.int).to(output.device) # removes indexes of false negatives 
                    #ipdb.set_trace()
                    neg = torch.index_select(output_normed, dim=0, index=idx_positives)
                    # replace removed samples from batch
                    neg = torch.cat((neg, torch.index_select(bank_normed, dim=0, index=idx_bank_negatives[torch.randperm(len(idx_bank_negatives))[:num_false_negatives]])), dim=0)
                negatives.append(neg)


        # TODO: so far selects top-1 nearest neighbor, try selecting farthest neighbor
        #index_nearest_neighbours = 
        #nearest_neighbours = torch.index_select(bank_normed, dim=0, index=index_nearest_neighbours)
        
        ipdb.set_trace()
        # stack all negative samples for each positive along row dimension
        negatives = torch.stack(negatives) # shape = (num_positives, num_negatives, embedding_size)
        
        return negatives, negatives



    def sinkhorn(self, Q, nmb_iters):
        if self.use_sinkhorn:
            with torch.no_grad():
                sum_Q = torch.sum(Q)
                Q /= sum_Q

                K, B = Q.shape

                if self.gpus != 0:
                    u = torch.zeros(K).cuda()
                    r = torch.ones(K).cuda() / K
                    c = torch.ones(B).cuda() / B
                else:
                    u = torch.zeros(K)
                    r = torch.ones(K) / K
                    c = torch.ones(B) / B

                for _ in range(nmb_iters):
                    u = torch.sum(Q, dim=1)

                    Q *= (r / u).unsqueeze(1)
                    Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

                return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
        return Q.t()


