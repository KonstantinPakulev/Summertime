import torch

from ignite.engine import Engine

"""
Loop modes
"""
MODE = 'mode'

TRAIN = "train"
VAL = "val"
TEST = "test"
VISUALIZE = 'visualize'
ANALYZE = 'analyze'


class Loop:

    def __init__(self, device, mode,
                 model, criterion_chain, optimizer_chain, loader):
        self.device = device
        self.mode = mode

        self.model = model
        self.criterion_chain = criterion_chain
        self.optimizer_chain = optimizer_chain
        self.loader = loader

        def iteration(engine, batch):
            if mode == TRAIN:
                self.model.train()

                # TODO. Add as option to config
                # with torch.autograd.detect_anomaly():
                endpoint = self.model(engine, batch)

                if endpoint is not None:
                    self.criterion_chain(engine, batch, endpoint)
                    self.optimizer_chain.step(engine, endpoint)

            elif mode in [VAL, ANALYZE]:
                self.model.eval()

                with torch.no_grad():
                    endpoint = self.model(engine, batch)

                    if endpoint is not None:
                        self.criterion_chain(engine, batch, endpoint)

            elif mode in [TEST, VISUALIZE]:
                self.model.eval()

                with torch.no_grad():
                    endpoint = self.model(engine, batch)

            else:
                raise Exception(f"Unknown mode {mode}")

            return batch, endpoint

        self.engine = Engine(iteration)

    def run(self, num_epochs, return_output=False):
        self.engine.run(self.loader, max_epochs=num_epochs)

        if return_output:
            return self.engine.state.output, self.engine.state.metrics
        else:
            return None