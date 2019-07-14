from torch.optim import lr_scheduler
from novelly.utils import build_from_config


class WarmupLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, burn_in_steps, after_scheduler=None, **kwargs):
        self.burn_in_steps = burn_in_steps
        if isinstance(after_scheduler, dict):
            after_scheduler = build_from_config(lr_scheduler, after_scheduler)
        self.after_scheduler = after_scheduler
        super().__init__(optimizer, **kwargs)

    def state_dict(self):
        state_dict = super().state_dict()
        if self.after_scheduler is not None:
            state_dict['after_scheduler'] = self.after_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict = state_dict.copy()
        if self.after_scheduler is not None:
            self.after_scheduler.load_state_dict(state_dict.pop('after_scheduler'))
        super().load_state_dict(state_dict)

    def get_lr(self):
        if self.last_epoch < self.burn_in_steps:
            return [lr * (self.last_epoch / self.burn_in_steps)**4
                    for lr in self.base_lrs]
        else:
            if self.after_scheduler is not None:
                return self.after_scheduler.get_lr()
            else:
                return self.base_lrs

    def step(self, epoch=None, metrics=None):
        super().step(epoch=epoch)
        if self.after_scheduler is not None:
            try:
                self.after_scheduler.step(epoch=epoch, metrics=metrics)
            except TypeError:
                self.after_scheduler.step(epoch=epoch)
