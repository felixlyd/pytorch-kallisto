import time
from datetime import timedelta

import torch

from criterion import KallistoLossFunc
from model import Kallisto, KallistoA
from optimizer import Optimizer
from option import Opt
from plot import Plot
from write_res import write_res


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    opt = Opt()
    opt.set_seed()
    args = opt.args
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    criterion = KallistoLossFunc(args.loss_table1, args.loss_table2)
    input_size = criterion.trans_nums
    inputs = torch.LongTensor(range(input_size)).to(device)
    model = Kallisto(input_size).to(device)
    alphas = 0
    optimizer = Optimizer(args, model.parameters())
    plot = Plot(args)
    plot.write_information()
    start_time = time.time()
    model.train()
    for epoch in range(args.epoch_num):
        optimizer.zero_grad()
        alphas = model(inputs)
        loss = criterion.nll_loss(alphas)
        criterion.backward()
        optimizer.update()
        optimizer.lr_decay()
        if epoch % 1000 == 0:
            print('Epoch [{}/{}]'.format(epoch, args.epoch_num))
            plot.write_loss_acc(loss.item(), epoch)
            plot.write_lr(optimizer.get_iter_lr(), epoch)
            plot.print_msg(loss.item(), epoch, get_time_dif(start_time))
    print("Time usage:", get_time_dif(start_time))
    print("EM loss:{:.8f}".format(criterion.em_loss.item()))
    plot.write_done()
    model.eval()
    counts = alphas.cpu().squeeze(1) * criterion.counts_sum
    write_res(counts, args.out, criterion.abundance_file)
