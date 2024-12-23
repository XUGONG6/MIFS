import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")


    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _cosine_simililarity(self, x, y):

        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def _get_correlated_mask(self):

        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))  #
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)

        return v

    def forward(self, z_ori, z_hard_pos, z_soft_pos, z_soft_neg):

        representations_ori_hp = torch.cat([z_ori, z_hard_pos], dim=0)
        representations_ori_sp = torch.cat([z_ori, z_soft_pos], dim=0)
        representations_ori_sn = torch.cat([z_ori, z_soft_neg], dim=0)
        representations_hp_sp = torch.cat([z_hard_pos, z_soft_pos], dim=0)
        representations_hp_sn = torch.cat([z_hard_pos, z_soft_neg], dim=0)
        representations_sp_sn = torch.cat([z_soft_pos, z_soft_neg], dim=0)

        similarity_matrix_ori_hp = self.similarity_function(representations_ori_hp, representations_ori_hp)
        similarity_matrix_ori_sp = self.similarity_function(representations_ori_sp, representations_ori_sp)
        similarity_matrix_ori_sn = self.similarity_function(representations_ori_sn, representations_ori_sn)
        similarity_matrix_hp_sp = self.similarity_function(representations_hp_sp, representations_hp_sp)
        similarity_matrix_hp_sn = self.similarity_function(representations_hp_sn, representations_hp_sn)
        similarity_matrix_sp_sn = self.similarity_function(representations_sp_sn, representations_sp_sn)

        l_pos_ori_sn = torch.diag(similarity_matrix_ori_sn, self.batch_size)
        r_pos_ori_sn = torch.diag(similarity_matrix_ori_sn, -self.batch_size)
        positives_ori_sn = torch.cat([l_pos_ori_sn, r_pos_ori_sn]).view(2 * self.batch_size, 1)

        l_pos_hp_sn = torch.diag(similarity_matrix_hp_sn, self.batch_size)
        r_pos_hp_sn = torch.diag(similarity_matrix_hp_sn, -self.batch_size)
        positives_hp_sn = torch.cat([l_pos_hp_sn, r_pos_hp_sn]).view(2 * self.batch_size, 1)

        l_pos_sp_sn = torch.diag(similarity_matrix_sp_sn, self.batch_size)
        r_pos_sp_sn = torch.diag(similarity_matrix_sp_sn, -self.batch_size)
        positives_sp_sn = torch.cat([l_pos_sp_sn, r_pos_sp_sn]).view(2 * self.batch_size, 1)

        l_pos_ori_hp = torch.diag(similarity_matrix_ori_hp, self.batch_size)
        r_pos_ori_hp = torch.diag(similarity_matrix_ori_hp, -self.batch_size)
        positives_ori_hp = torch.cat([l_pos_ori_hp, r_pos_ori_hp]).view(2 * self.batch_size, 1)
        negatives_ori_hp = similarity_matrix_ori_hp[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        l_pos_ori_sp = torch.diag(similarity_matrix_ori_sp, self.batch_size)
        r_pos_ori_sp = torch.diag(similarity_matrix_ori_sp, -self.batch_size)
        positives_ori_sp = torch.cat([l_pos_ori_sp, r_pos_ori_sp]).view(2 * self.batch_size, 1)
        negatives_ori_sp = similarity_matrix_ori_sp[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        l_pos_hp_sp = torch.diag(similarity_matrix_hp_sp, self.batch_size)
        r_pos_hp_sp = torch.diag(similarity_matrix_hp_sp, -self.batch_size)
        positives_hp_sp = torch.cat([l_pos_hp_sp, r_pos_hp_sp]).view(2 * self.batch_size, 1)
        negatives_hp_sp = similarity_matrix_hp_sp[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits_ori_hp = torch.cat((positives_ori_hp, negatives_ori_hp, positives_ori_sn), dim=1)
        logits_ori_hp /= self.temperature

        logits_ori_sp = torch.cat((positives_ori_sp, negatives_ori_sp, positives_ori_sn), dim=1)
        logits_ori_sp /= self.temperature

        logits_hp_sp = torch.cat((positives_hp_sp, negatives_hp_sp, positives_hp_sn, positives_sp_sn), dim=1)
        logits_hp_sp /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()    # 2 * batch
        loss_ori_hp = self.criterion(logits_ori_hp, labels)
        loss_ori_sp = self.criterion(logits_ori_sp, labels)
        logits_hp_sp = self.criterion(logits_hp_sp, labels)

        return loss_ori_hp / (2 * self.batch_size), loss_ori_sp / (2 * self.batch_size), logits_hp_sp / (2 * self.batch_size)


class Weight_NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(Weight_NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature          # 0.1
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)     # true
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (2B, 1, D)
        # y shape: (1, D, 2B)
        # v shape: (2B, 2B)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (2B, 1, D)
        # y shape: (1, D, 2B)
        # v shape: (2B, 2B)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, z_ori, z_hard_pos, z_soft_pos, z_soft_neg):

        representations_ori_hp_averge = torch.mean(torch.stack((z_ori, z_hard_pos)), dim=0)   # Bx D  对三维矩阵的dim=0求平均，
        representations_ori_sp_averge = torch.mean(torch.stack((z_ori, z_soft_pos)), dim=0)   # B x D
        representations_ori_sn_averge = torch.mean(torch.stack((z_ori, z_soft_neg)), dim=0)   # B x D
        representations_hp_sp_averge = torch.mean(torch.stack((z_hard_pos, z_soft_pos)), dim=0)   # Bx D
        representations_hp_sn_averge = torch.mean(torch.stack((z_hard_pos, z_soft_neg)), dim=0)   # B x D
        representations_sp_sn_averge = torch.mean(torch.stack((z_soft_pos, z_soft_neg)), dim=0)   # Bx D

        #  下面操作，原始数据减去相应均值，然后按照列(下)拼接，维度2B*D
        representations_ori_hp = torch.cat([z_ori-representations_ori_hp_averge, z_hard_pos-representations_ori_hp_averge], dim=0)  # 锚点-硬正 B+B x D
        representations_ori_sp = torch.cat([z_ori-representations_ori_sp_averge, z_soft_pos-representations_ori_sp_averge], dim=0)  # 锚点-软正 B+B x D
        representations_ori_sn = torch.cat([z_ori-representations_ori_sn_averge, z_soft_neg-representations_ori_sn_averge], dim=0)  # 锚点-软负 B+B x D
        representations_hp_sp = torch.cat([z_hard_pos-representations_hp_sp_averge, z_soft_pos-representations_hp_sp_averge], dim=0)  # 硬正-软正 B+B x D
        representations_hp_sn = torch.cat([z_hard_pos-representations_hp_sn_averge, z_soft_neg-representations_hp_sn_averge], dim=0)  # 硬正-软负 B+B x D
        representations_sp_sn = torch.cat([z_soft_pos-representations_sp_sn_averge, z_soft_neg-representations_sp_sn_averge], dim=0)  # 软正-软负 B+B x D

        # 求 弦相似度，得到相似度矩阵。 self.similarity_function（x, y, v）
        # x shape: (2B, 1, D)
        # y shape: (1, D, 2B)
        # v shape: (2B, 2B)
        similarity_matrix_ori_hp = self.similarity_function(representations_ori_hp, representations_ori_hp)     # 2B x 2B
        similarity_matrix_ori_sp = self.similarity_function(representations_ori_sp, representations_ori_sp)     # 2B x 2B
        similarity_matrix_ori_sn = self.similarity_function(representations_ori_sn, representations_ori_sn)     # 2B x 2B
        similarity_matrix_hp_sp = self.similarity_function(representations_hp_sp, representations_hp_sp)        # 2B x 2B
        similarity_matrix_hp_sn = self.similarity_function(representations_hp_sn, representations_hp_sn)        # 2B x 2B
        similarity_matrix_sp_sn = self.similarity_function(representations_sp_sn, representations_sp_sn)        # 2B x 2B

        l_pos_ori_sn = torch.diag(similarity_matrix_ori_sn, self.batch_size)
        r_pos_ori_sn = torch.diag(similarity_matrix_ori_sn, -self.batch_size)
        positives_ori_sn = torch.cat([l_pos_ori_sn, r_pos_ori_sn]).view(2 * self.batch_size, 1)


        l_pos_hp_sn = torch.diag(similarity_matrix_hp_sn, self.batch_size)    # right positive
        r_pos_hp_sn = torch.diag(similarity_matrix_hp_sn, -self.batch_size)   # left positive
        positives_hp_sn = torch.cat([l_pos_hp_sn, r_pos_hp_sn]).view(2 * self.batch_size, 1)

        l_pos_sp_sn = torch.diag(similarity_matrix_sp_sn, self.batch_size)    # right positive
        r_pos_sp_sn = torch.diag(similarity_matrix_sp_sn, -self.batch_size)   # left positive
        positives_sp_sn = torch.cat([l_pos_sp_sn, r_pos_sp_sn]).view(2 * self.batch_size, 1)

        l_pos_ori_hp = torch.diag(similarity_matrix_ori_hp, self.batch_size)    # right positive
        r_pos_ori_hp = torch.diag(similarity_matrix_ori_hp, -self.batch_size)   # left positive
        positives_ori_hp = torch.cat([l_pos_ori_hp, r_pos_ori_hp]).view(2 * self.batch_size, 1)
        negatives_ori_hp = similarity_matrix_ori_hp[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # 锚点-软正
        l_pos_ori_sp = torch.diag(similarity_matrix_ori_sp, self.batch_size)
        r_pos_ori_sp = torch.diag(similarity_matrix_ori_sp, -self.batch_size)
        positives_ori_sp = torch.cat([l_pos_ori_sp, r_pos_ori_sp]).view(2 * self.batch_size, 1)     # batch * 1
        negatives_ori_sp = similarity_matrix_ori_sp[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # 硬正-软正
        l_pos_hp_sp = torch.diag(similarity_matrix_hp_sp, self.batch_size)
        r_pos_hp_sp = torch.diag(similarity_matrix_hp_sp, -self.batch_size)
        positives_hp_sp = torch.cat([l_pos_hp_sp, r_pos_hp_sp]).view(2 * self.batch_size, 1)     # batch * 1
        negatives_hp_sp = similarity_matrix_hp_sp[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)


        # ?????????????
        logits_ori_hp = torch.cat((positives_ori_hp, negatives_ori_hp, positives_ori_sn), dim=1)   # batch * 3
        logits_ori_hp /= self.temperature

        logits_ori_sp = torch.cat((positives_ori_sp, negatives_ori_sp, positives_ori_sn), dim=1)   # batch * 3
        logits_ori_sp /= self.temperature

        logits_hp_sp = torch.cat((positives_hp_sp, negatives_hp_sp, positives_hp_sn, positives_sp_sn), dim=1)   # batch * 3
        logits_hp_sp /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()    # 2 * batch
        loss_ori_hp = self.criterion(logits_ori_hp, labels)
        loss_ori_sp = self.criterion(logits_ori_sp, labels)
        logits_hp_sp = self.criterion(logits_hp_sp, labels)

        return loss_ori_hp / (2 * self.batch_size), loss_ori_sp / (2 * self.batch_size), logits_hp_sp / (2 * self.batch_size)
