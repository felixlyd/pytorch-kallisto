import argparse
import random

import numpy as np
import torch

from common import OPTIMIZERS, LR_SCHEDULERS, ACTIVATIONS


class Opt:
    def __init__(self):
        self.parser =None
        self.args = None
        self.init_state = False
        if not self.init_state:
            self.build()
            self.init_state = True
        self.print_opt()

    def build(self):
        parser = argparse.ArgumentParser(
            description="Pytorch-Kallisto",
            add_help=True,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.add_args(parser)
        self.parser = parser
        self.args = parser.parse_args()

    def add_args(self, parser):
        parser.add_argument('--loss_table1', default='resources/data',
                            help="loss table file1. 'pseudoalignments_x.tsv'")
        parser.add_argument('--loss_table2', default='resources/data',
                            help="loss table file2.  'abundance.tsv'")
        parser.add_argument('--log', default="resources/log", help="path to the log folder to record information.")
        parser.add_argument('--epoch_num', type=int, default=10000, help='epoch size.')
        parser.add_argument('--optimizer', type=str, default='Adam', choices=OPTIMIZERS,
                            help="chooses which optimizer to use. ")
        parser.add_argument('--active', type=str, default='NULL', choices=ACTIVATIONS,
                            help="chooses which activations to use.")
        parser.add_argument('--lr', type=float, default=0.01, help="initial learning rate.")
        parser.add_argument('--lr_scheduler', type=str, choices=LR_SCHEDULERS,
                            help="chooses which lr_scheduler to use.")
        parser.add_argument('--plot', action="store_true", help='if specified, plot the logs powered by tensorboard.')
        parser.add_argument('--seed', type=int, default=24, help="random seed.")
        parser.add_argument('--out', type=str,  help="out file.")

    def print_opt(self):
        message = ""
        message = message + "-" * 20 + "Options" + "-" * 20 + "\n"
        for arg in vars(self.args):
            arg_name = arg
            arg_value = vars(self.args)[arg_name]
            default_value = self.parser.get_default(arg_name)
            remark = ""
            if arg_value != default_value:
                remark = '\t(default: {})\t'.format(default_value)
            message = message + "{}: {}{}\n".format(arg_name, arg_value, remark)
        message = message + "-" * 20 + "End" + "-" * 20
        print(self.parser.description)
        print(message)

    def set_seed(self):
        seed = self.args.seed
        if self.args.seed is None:
            seed = 24
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
