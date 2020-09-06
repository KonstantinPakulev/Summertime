from abc import ABC

import torch.nn as nn

import Net.source.core.experiment as exp

from Net.source.core.ignite_metrics import AveragePeriodicMetric, KeyTransformer

"""
Loss keys
"""

LOSS = 'loss'


class ModuleWrapper(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, engine, batch, endpoint, bundle):
        raise NotImplementedError


class AttachableModelWrapper(ModuleWrapper, ABC):

    def attach(self, engine, bundle):
        pass


class CriterionChain:

    def __init__(self, criterion_wrappers):
        self.criterion_wrappers = criterion_wrappers

    def __call__(self, engine, batch, endpoint):
        final_loss = 0
        bundle = {}

        for c_w in self.criterion_wrappers:
            loss, endpoint, bundle = c_w(engine, batch, endpoint, bundle)

            final_loss = final_loss + loss

        endpoint[LOSS] = final_loss

    def attach(self, engine, bundle):
        loss_log_iter = bundle.get(exp.LOSS_LOG_ITER)
        AveragePeriodicMetric(KeyTransformer(LOSS), loss_log_iter).attach(engine, LOSS)

        for c_w in self.criterion_wrappers:
            c_w.attach(engine, bundle)


class OptimizerChain:

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, engine, endpoint):
        loss = endpoint[LOSS]

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        for param_group in self.optimizer.param_groups:
            print(param_group['lr'], flush=True)
            break

    def state_dict(self):
        # TODO. Need more advanced save/loading routines in case several optimizers are used
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)


class ModelContainer(nn.Module):

    def __init__(self, device, model_wrappers):
        super().__init__()
        self.device = device
        self.container = nn.ModuleList(model_wrappers)

    def forward(self, engine, batch):
        bundle = {}
        endpoint = {}

        # TODO. Insert time measurement routine

        for m_w in self.container:
            endpoint, bundle = m_w(engine, batch, endpoint, bundle)

        return endpoint

    def attach(self, engine, bundle):
        # TODO. Maybe add time measurements to log?
        pass
