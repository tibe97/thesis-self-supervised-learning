""" Nearest Neighbour Memory Bank Module with method for hard-negative sampling"""

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import time
import copy
import ipdb
import numpy as np
from lightly.loss.memory_bank import MemoryBankModule


class MyNNMemoryBankModule(MemoryBankModule):
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
        super(MyNNMemoryBankModule, self).__init__(size)
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
                num_nn: int,
                update: bool = False):
        """Returns nearest neighbour of output tensor from memory bank which belongs to the same cluster
        as the reference

        Args:
            output: The torch tensor for which you want the nearest neighbour
            update: If `True` updated the memory bank by adding output to it

        """
        
        output, bank = super(MyNNMemoryBankModule, self).forward(output, update=update)
        bank = bank.to(output.device).t()
        #ipdb.set_trace()
        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        #compute cluster assignments
        #start_time = time.time()
        with torch.no_grad():
            cluster_scores = torch.mm(torch.cat((output_normed, bank_normed)), self.model.prototypes_layer.weight.t())
            q = torch.exp(cluster_scores / self.epsilon).t()
            q = self.get_assignments(q, self.sinkhorn_iterations)
            
            # separate assignments for positives and for negatives 
            q_batch = q[:output.shape[0]]
            q_bank = q[output.shape[0]:]
            # transform soft assignment into hard assignments to get cluster
            clusters_batch = torch.argmax(q_batch, dim=1)
            clusters_bank = torch.argmax(q_bank, dim=1)
        #end_time = time.time()
        #print("Compute cluster assignments: {}".format(end_time-start_time))

        # TODO: 1. pick random negatives; 2. batch + hard negatives as negatives
        negatives = []
        similarity_matrix_pos = torch.einsum("nd,md->nm", output_normed, bank_normed)
        similarity_matrix_neg = copy.deepcopy(similarity_matrix_pos)

        #start_time = time.time()
       
        # efficient implementation of the loops with scatter_() function
        for i in range(similarity_matrix_pos.shape[0]): # for each positive sample
            row_pos = similarity_matrix_pos[i]
            row_neg = similarity_matrix_neg[i]
            p_cluster = clusters_batch[i]
            mask_indices = torch.where(clusters_bank==p_cluster)[0]
            row_pos.scatter_(0, mask_indices, 10, reduce='add')
            row_neg.scatter_(0, mask_indices, -10, reduce='add')
            # We take the indices of the most similar negatives. Try with random negatives
            idx_negatives = torch.topk(row_neg, num_nn, dim=0, largest=True).indices

            if not self.false_neg_remove:
                negatives.append(torch.index_select(bank_normed, dim=0, index=idx_negatives))
                
            elif self.soft_neg: # Hard negatives + batch without false negatives
                negatives.append(torch.index_select(bank_normed, dim=0, index=idx_negatives))
                mask_positives = torch.where(clusters_batch==p_cluster)[0]
                num_false_negatives = mask_positives.shape[0]
                neg = output_normed
                if num_false_negatives > 0:
                    idx_positives = torch.Tensor(list(set(range(output.shape[0])) - set(mask_positives.tolist()))).to(torch.int).to(output.device) # removes indexes of false negatives 
                    #ipdb.set_trace()
                    neg = torch.index_select(output_normed, dim=0, index=idx_positives)
                    # replace removed samples from batch
                    neg = torch.cat((neg, torch.index_select(bank_normed, dim=0, index=idx_negatives[:num_false_negatives])), dim=0)
                negatives.append(neg)
            else:
                # False Negatives removal from batch
                # remove false negatives from batch (i.e. positives) and replace them with samples
                # from memory bank
                mask_positives = torch.where(clusters_batch==p_cluster)[0]
                num_false_negatives = mask_positives.shape[0]
                neg = output_normed
                if num_false_negatives > 0:
                    idx_positives = torch.Tensor(list(set(range(output.shape[0])) - set(mask_positives.tolist()))).to(torch.int).to(output.device) # removes indexes of false negatives 
                    #ipdb.set_trace()
                    neg = torch.index_select(output_normed, dim=0, index=idx_positives)
                    # replace removed samples from batch
                    neg = torch.cat((neg, torch.index_select(bank_normed, dim=0, index=idx_negatives[:num_false_negatives])), dim=0)
                negatives.append(neg)


        # TODO: so far selects top-1 nearest neighbor, try selecting farthest neighbor
        index_nearest_neighbours = torch.argmax(similarity_matrix_pos, dim=1)
        nearest_neighbours = torch.index_select(bank_normed, dim=0, index=index_nearest_neighbours)
        
        #end_time = time.time()
        #print("Sampling time of positives and negatives: {}".format(end_time-start_time))
        
        # stack all negative samples for each positive along row dimension
        negatives = torch.stack(negatives) # shape = (num_positives, num_negatives, embedding_size)
        
        return nearest_neighbours, negatives, q_batch



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


