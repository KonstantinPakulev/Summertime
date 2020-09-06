import itertools
import pandas as pd

from enum import Enum

from ignite.metrics import Metric
from ignite.engine import Events, State

import torch


class KeyTransformer:

    def __init__(self, key):
        self.key = key

    def __call__(self, output):
        batch, endpoint = output

        if self.key in endpoint:
            return endpoint[self.key]
        else:
            return batch.get(self.key)


class AveragePeriodicMetric(Metric):

    def __init__(self, output_transform, log_interval=None):

        self.log_interval = log_interval

        self.value = 0
        self.iter_counter = 0

        super().__init__(output_transform)

    def reset(self):
        if self.log_interval is None or self.iter_counter % self.log_interval == 0:
            self.value = 0
            self.iter_counter = 0

    def update(self, output):
        if output is not None:
            self.value += output
            self.iter_counter += 1

    def compute(self):
        if self.iter_counter == 0:
            return None
        else:
            return self.value / self.iter_counter

    def attach(self, engine, name):
        if self.log_interval is not None:
            engine.add_event_handler(Events.ITERATION_STARTED, self.started)  # Reset if needed
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)  # Update

            # Create periodic event
            custom_state = f"{name}_iteration"
            periodic_event_name = f"{custom_state.upper()}_FINISHED"
            setattr(self, "Events", Enum("Events", periodic_event_name))

            for e in self.Events:
                State.event_to_attr[e] = custom_state

            periodic_event = getattr(self.Events, periodic_event_name)

            def on_periodic_event(engine_p):
                if engine_p.state.iteration % self.log_interval == 0:
                    engine_p.fire_event(periodic_event)

            engine.register_events(*self.Events)
            engine.add_event_handler(Events.ITERATION_COMPLETED, on_periodic_event)

            engine.add_event_handler(periodic_event, self.completed, name)  # Compute
        else:
            super().attach(engine, name)


class DetailedMetric(Metric):

    def __init__(self, output_transform, num_cat):
        self.num_cat = num_cat
        self.log = [{} for _ in range(num_cat)]

        super().__init__(output_transform)

    def reset(self):
        for i in range(self.num_cat):
            for _k in self.log[i].keys():
                del self.log[i][_k][:]

    def update(self, output):
        for i in range(self.num_cat):
            for _k, _v in output[i].items():
                if _k in self.log[i]:
                    self.log[i][_k].append(_v)
                else:
                    self.log[i][_k] = [_v]

    def compute(self):
        results = []

        for i in range(self.num_cat):
            for _k, _v in self.log[i].items():
                if torch.is_tensor(self.log[i][_k][0]):
                    self.log[i][_k] = torch.cat(self.log[i][_k], dim=0).cpu().numpy().tolist()

                elif isinstance(self.log[i][_k][0], list):
                    self.log[i][_k] = list(itertools.chain(*self.log[i][_k]))

            results.append(pd.DataFrame(data=self.log[i]))

        return results
