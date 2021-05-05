import json
import os
import time

from tensorboardX import SummaryWriter


class Plot:
    def __init__(self, opt):
        self.opt = opt
        self.lr = opt.lr
        self.plot = opt.plot
        self.log = opt.log
        self.msg = 'Iter: {0:>6},  Loss: {1:.8f},  Time: {2}'
        if self.plot:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.log, "Kallisto_" + opt.optimizer + "_"
                                     + time.strftime('%m-%d_%H.%M', time.localtime())))
        else:
            self.writer = None

    def write_done(self):
        if self.plot:
            self.writer.close()

    def write_loss_acc(self, loss_, iter_):
        if self.plot:
            self.writer.add_scalar("loss", loss_, iter_)
        else:
            pass

    def print_msg(self, loss_, iter_, time_dif):
        print(self.msg.format(iter_, loss_, time_dif))

    def write_information(self):
        if self.plot:
            information = json.dumps(vars(self.opt))
            self.writer.add_text("Pytorch-Kallisto/Opts", information)

    def write_lr(self, lr_, iter_):
        self.writer.add_scalar("learning_rate", lr_, iter_)
