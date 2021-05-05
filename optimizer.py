import torch.optim as optim

class Optimizer:
    def __init__(self, opt, params):
        self.optimizer_name = opt.optimizer
        self.lr_scheduler_name = opt.lr_scheduler
        self.optimizer = None
        self.lr_scheduler = None
        self.lr = opt.lr
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.gamma = opt.gamma

        self._init_optimizer(params)
        self._init_lr_scheduler()

    def _init_optimizer(self, params):
        if self.optimizer_name == "Adam":
            self.optimizer = optim.Adam(params, lr=self.lr, betas=(self.beta1, self.beta2))

    def _init_lr_scheduler(self):
        if self.lr_scheduler_name is None:
            self.lr_scheduler = None
        elif self.lr_scheduler_name == "StepLR":
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=self.gamma)
        elif self.lr_scheduler_name == "ExponentialLR":
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)
        elif self.lr_scheduler_name == "CosineAnnealingLR":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update(self):
        self.optimizer.step()

    def lr_decay(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def get_iter_lr(self):
        return self.optimizer.param_groups[0]['lr']