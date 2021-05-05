import torch
from tqdm import tqdm
import numpy as np


class KallistoLossFunc:
    def __init__(self, ecs_file, abundance_file):
        self.labels = None
        self.eff_lens = None
        self.counts = None
        self.names = None
        self.trans_nums = 0

        self.ecs_file = ecs_file
        self.abundance_file = abundance_file
        self.load_criterion_table()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.set_device()

    def load_criterion_table(self):
        print("Loading ecs...")
        inputs = []
        label_list = []
        counts = np.array([]).astype("float")
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
        counts = torch.FloatTensor(counts).reshape(1, -1)
        self.counts = counts / torch.sum(counts)
        idx_list = []
        for i, label in enumerate(label_list):
            idx_list.extend([[i, x] for x in label])
        index = torch.LongTensor(idx_list)
        values = torch.FloatTensor([1.] * len(idx_list))
        self.labels = torch.sparse_coo_tensor(index.t(), values, torch.Size([len(label_list), len(inputs)]),
                                              dtype=torch.float)
        print("Loading eff_lens...")
        eff_lens = np.array([]).astype("float")
        gene_names = []
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
                    eff_lens = np.append(eff_lens, float(eff_length))
                    gene_names.append(gene_name)
        self.eff_lens = torch.FloatTensor(eff_lens).reshape(-1, 1)
        self.names = gene_names

    def set_device(self):
        self.labels = self.labels.to(self.device)
        self.eff_lens = self.eff_lens.to(self.device)
        self.counts = self.counts.to(self.device)

    def nll_loss(self, alphas):
        # alphas.size [len(self.trans_nums), 1]
        x = torch.div(alphas, self.eff_lens)
        x = torch.sparse.mm(self.labels, x)
        x = torch.log(x)
        x = torch.mm(self.counts, x)
        x = -x
        return x


if __name__ == '__main__':
    loss_func = KallistoLossFunc("resources/data/ath/pseudoalignments_ath.tsv",
                                 "resources/data/ath/abundance.tsv")
    num = loss_func.trans_nums
    alpha = torch.FloatTensor([1. / num] * num).reshape(-1, 1).requires_grad_(True)
    alpha = alpha.to("cuda:0")
    loss = loss_func.nll_loss(alpha)
