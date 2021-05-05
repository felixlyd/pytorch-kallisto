import torch
from tqdm import tqdm
import numpy as np
from decimal import Decimal


class KallistoLossFunc:
    def __init__(self, ecs_file, abundance_file):
        self.labels = None
        self.eff_lens = None
        self.counts = None
        self.em_counts = None
        self.names = None
        self.trans_nums = 0
        self.counts_sum = 0
        self.em_loss = 0.0

        self.ecs_file = ecs_file
        self.abundance_file = abundance_file
        self.load_criterion_table()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.set_device()
        self.loss = None
        self.em_nll_loss()

    def load_criterion_table(self):
        print("Loading ecs...")
        inputs = []
        label_list = []
        counts = []
        with open(self.ecs_file, 'r') as f:
            for line in tqdm(f):
                line_str = line.strip()
                if not line_str:
                    break
                if "EC	Count	Transcript" in line:
                    # 首行不处理
                    pass
                else:
                    idx, count, label = line_str.split("\t")
                    label = [int(x) for x in label.split(",")]
                    if len(label) == 1:
                        inputs.append(label[0])
                    label_list.append(label)
                    counts = np.append(counts, int(count))
        self.trans_nums = len(inputs)
        counts = torch.DoubleTensor(counts).reshape(1, -1)
        self.counts_sum = torch.sum(counts)
        self.counts = counts / self.counts_sum
        idx_list = []
        for i, label in enumerate(label_list):
            idx_list.extend([[i, x] for x in label])
        index = torch.LongTensor(idx_list)
        values = torch.DoubleTensor([1.] * len(idx_list))
        self.labels = torch.sparse_coo_tensor(index.t(), values, torch.Size([len(label_list), len(inputs)]),
                                              dtype=torch.double)
        print("Loading eff_lens...")
        eff_lens = []
        gene_names = []
        em_counts = []
        with open(self.abundance_file, 'r') as f:
            for line in tqdm(f):
                line_str = line.strip()
                if not line_str:
                    break
                if "target_id	length	eff_length	est_counts	tpm" in line:
                    # 首行不处理
                    pass
                else:
                    gene_name, length, eff_length, est_counts, tpm = line_str.split("\t")
                    eff_lens.append(eff_length)
                    em_counts.append(est_counts)
                    gene_names.append(gene_name)
        em_counts = np.array(em_counts).astype("float64")
        eff_lens = np.array(eff_lens).astype("float64")
        self.eff_lens = torch.DoubleTensor(eff_lens).reshape(-1, 1)
        self.names = gene_names
        em_counts = torch.DoubleTensor(em_counts).reshape(-1, 1)
        self.em_counts = em_counts / self.counts_sum

    def set_device(self):
        self.labels = self.labels.to(self.device)
        self.eff_lens = self.eff_lens.to(self.device)
        self.counts = self.counts.to(self.device)
        self.em_counts = self.em_counts.to(self.device)

    def nll_loss(self, alphas):
        # alphas.size [len(self.trans_nums), 1]
        self.loss = torch.div(alphas, self.eff_lens)
        self.loss = torch.sparse.mm(self.labels, self.loss)
        self.loss = torch.log(self.loss)
        self.loss = torch.mm(self.counts, self.loss)
        self.loss = -self.loss
        return self.loss

    def em_nll_loss(self):
        self.em_loss = torch.div(self.em_counts, self.eff_lens)
        self.em_loss = torch.sparse.mm(self.labels, self.em_loss)
        self.em_loss = torch.log(self.em_loss)
        self.em_loss = torch.where(torch.isinf(self.em_loss), torch.full_like(self.em_loss, 0), self.em_loss)
        self.em_loss = torch.mm(self.counts, self.em_loss)
        self.em_loss = -self.em_loss

    def backward(self):
        self.loss.backward()


if __name__ == '__main__':
    loss_func = KallistoLossFunc("resources/data/ath/pseudoalignments_ath.tsv",
                                 "resources/data/ath/abundance.tsv")
    num = loss_func.trans_nums
    alpha = torch.DoubleTensor([1. / num] * num).reshape(-1, 1).requires_grad_(True)
    alpha = alpha.to("cuda:0")
    loss = loss_func.nll_loss(alpha)
