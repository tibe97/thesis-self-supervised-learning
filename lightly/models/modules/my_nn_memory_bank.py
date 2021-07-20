""" Nearest Neighbour Memory Bank Module with method for hard-negative sampling"""

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import time
import ipdb
import copy
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
                use_sinkhorn: bool = False):
        super(MyNNMemoryBankModule, self).__init__(size)
        self.model = model
        self.epsilon = epsilon #division coefficient for cluster assignemnt computation
        self.get_assignments = self.sinkhorn
        self.sinkhorn_iterations = sinkhorn_iterations
        self.gpus = gpus
        self.use_sinkhorn = use_sinkhorn
        

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
        start_time = time.time()
        output, bank = super(MyNNMemoryBankModule, self).forward(output, update=update)
        device = output.device
        bank = bank.to(output.device).t()
        end_time = time.time()
        print("Retrieve bank and move to GPU: {}".format(end_time-start_time))

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        #compute cluster assignments
        start_time = time.time()
        with torch.no_grad():
            """
            w = self.model.prototypes_layer.weight.data.clone()
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes_layer.weight.copy_(w)
            """
            cluster_scores = torch.mm(torch.cat((output_normed, bank_normed)), self.model.prototypes_layer.weight.t())

            q = torch.exp(cluster_scores / self.epsilon).t()
            q = self.get_assignments(q, self.sinkhorn_iterations)
            
            # separate assignments for positives from negatives 
            q_positives = q[:output.shape[0]]
            q_negatives = q[output.shape[0]:]
            # transform soft assignment into hard assignments to get cluster
            positive_clusters = torch.argmax(q_positives, dim=1)
            negative_clusters = torch.argmax(q_negatives, dim=1)
        end_time = time.time()
        print("Compute cluster assignments: {}".format(end_time-start_time))


        sim_negatives = []
        similarity_matrix_pos = torch.einsum("nd,md->nm", output_normed, bank_normed)
        similarity_matrix_neg = copy.deepcopy(similarity_matrix_pos)

        start_time = time.time()
        """
        for i in range(similarity_matrix_pos.shape[0]): # for each positive sample
            row_pos = similarity_matrix_pos[i]
            row_neg = similarity_matrix_neg[i]
            p_cluster = positive_clusters[i]
            mask_indices = torch.where(negative_clusters==p_cluster)[0]
            for idx in mask_indices:
                row_pos[idx] += 10 # to make sure we select among the same cluster
                row_neg[idx] -= 10 # to make sure we don't select from the same cluster
            sim_nearest_neighbours = torch.topk(row_neg, num_nn, dim=0).values # take the similarity score
            sim_negatives.append(sim_nearest_neighbours)
        index_nearest_neighbours = torch.argmax(similarity_matrix_pos, dim=1)
        nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours)
        
        
        """
        # efficient implementation of the loops
        for i in range(similarity_matrix_pos.shape[0]): # for each positive sample
            row_pos = similarity_matrix_pos[i]
            row_neg = similarity_matrix_neg[i]
            p_cluster = positive_clusters[i]
            mask_indices = torch.where(negative_clusters==p_cluster)[0]
            row_pos.scatter_(1, mask_indices, torch.cuda.FloatTensor(10), reduce='add')
            row_neg.scatter_(1, mask_indices, torch.cuda.FloatTensor(-10), reduce='add')
            sim_nearest_neighbours = torch.topk(row_neg, num_nn, dim=0).values # take the similarity score
            sim_negatives.append(sim_nearest_neighbours)
        index_nearest_neighbours = torch.argmax(similarity_matrix_pos, dim=1)
        nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours)
        
        end_time = time.time()
        print("Sample positives and negatives: {}".format(end_time-start_time))

        # stack all negative similarities for each positive along row dimension
        sim_negatives = torch.stack(sim_negatives) # (num_positives, num_negatives)
        

        return nearest_neighbours, sim_negatives, q_positives



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


class HardNegativeMemoryBankModule(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation with hard negative mining.
       Positives comnputed as in NNCLR

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
                use_sinkhorn: bool = False):
        super(HardNegativeMemoryBankModule, self).__init__(size)
        self.model = model
        self.epsilon = epsilon #division coefficient for cluster assignemnt computation
        self.get_assignments = self.sinkhorn
        self.sinkhorn_iterations = sinkhorn_iterations
        self.gpus = gpus
        self.use_sinkhorn = use_sinkhorn
 
        

    def forward(self,
                output: torch.Tensor,
                update: bool = False):
        """Returns nearest neighbour of output tensor from memory bank

        Args:
            output: The torch tensor for which you want the nearest neighbour
            update: If `True` updated the memory bank by adding output to it

        """
        output, bank = super(HardNegativeMemoryBankModule, self).forward(output, update=update)
        bank = bank.to(output.device).t()

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)


        similarity_matrix = torch.einsum("nd,md->nm", output_normed, bank_normed)
        index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
        nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours)

        return nearest_neighbours


    def sample_negatives(self,
                output: torch.Tensor,
                cluster_scores_positives: torch.Tensor,
                num_nn: int,
                update: bool = False):
        """Returns first NUM_NN nearest neighbour of output tensor from memory bank and take them as negatives
        only if they belong to a different class than the positive one.
        For each positive loop the top nearest neighbors and take the first NUM_NN neighbors such that
        they belong to a different class (hard negatives). We basically want to take the negatives from the closest clusters
        but in order to check that they belong to close clusters we just use the similarity as index. We then discard the
        samples with the same cluster assignment as the positive one.

        ALGORITHM:
        1. Compute the cluster assignement for each sample in the bank and for each positive.
        2. Compute similarity matrix between positives and samples in the memory bank.
           Each row gives the similarity score between each positive and all the samples in the bank.
        3. Loop each positive (row of similarity matrix) and get first NUM_NN nearest neighbors in the bank 
           such that the cluster assignment is different from the positive. 
        4. Return a 3D tensor of size (num_positives, emb_dim, num_negatives)



        Args:
            model: The model for computing clusters
            output: The torch tensor for which you want the nearest neighbour
            cluster_scores_positives: scores from the model output to compute cluster assignemnt
            num_nn: number of nearest neighbors to sample as negatives for each positive. Should be the size of the batch.
            update: If `True` updated the memory bank by adding output to it

        """
        
        output, bank = super(HardNegativeMemoryBankModule, self).forward(output, update=update)
        bank = bank.to(output.device).t()
        

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        # compute cluster assignemnts for positives and bank (from SwAV code)
        # compute scores for bank and concatenate with positive scores already computed
        # check if "bank_normed" should get trasposed
        # concatenate positive scores with computed negative scores for clustering
        with torch.no_grad():
            cluster_scores = torch.cat((cluster_scores_positives, torch.mm(bank_normed, self.model.prototypes_layer.weight.t())))

            q = torch.exp(cluster_scores / self.epsilon).t()
            q = self.get_assignments(q, self.sinkhorn_iterations)
            
            # separate assignments for positives from negatives 
            q_positives = q[:cluster_scores_positives.shape[0]]
            q_negatives = q[cluster_scores_positives.shape[0]:]
            # transform soft assignment into hard assignments to get cluster
            positive_clusters = torch.argmax(q_positives, dim=1)
            negative_clusters = torch.argmax(q_negatives, dim=1)

        sim_negatives = []
    
        similarity_matrix = torch.einsum("nd,md->nm", output_normed, bank_normed) # between positives and memory bank
        for i in range(similarity_matrix.shape[0]): # for each positive sample
            row = similarity_matrix[i]
            #mask out samples with the same cluster assignment by putting -1 as corresponding value
            p_cluster = positive_clusters[i]
            mask_indices = torch.where(negative_clusters==p_cluster)[0]
            for idx in mask_indices:
                row[idx] = -10 # to make sure we don't select them
            sim_nearest_neighbours = torch.topk(row, num_nn, dim=0).values # take the similarity score
            sim_negatives.append(sim_nearest_neighbours)

            #index_nearest_neighbours = torch.topk(row, num_nn, dim=0).indices # take indices of top K most similar
            #negatives.append(torch.index_select(bank, dim=0, index=index_nearest_neighbours))

        # stack all negative similarities for each positive along row dimension
        sim_negatives = torch.stack(sim_negatives) # (num_positives, num_negatives)
        #end_time = time.time()
        #print("Time Sample Negative: {}".format(end_time-start_time))

        return output, sim_negatives, q_positives



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