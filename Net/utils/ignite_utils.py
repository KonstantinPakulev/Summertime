from ignite.metrics import Metric
from ignite.engine import Events


class AverageMetric(Metric):

    def __init__(self, output_transform, log_interval=None):

        self.log_interval = log_interval

        if self.log_interval is not None and self.log_interval == 0:
            raise ValueError("Log interval can not be equal to zero")

        self.value = 0
        self.iter_counter = 0

        super().__init__(output_transform)

    def reset(self):
        if self.iter_counter != 0 and (self.log_interval is None or self.iter_counter % self.log_interval == 0):
            self.value = 0
            self.iter_counter = 0

    def update(self, output):
        self.value += output
        self.iter_counter += 1

    def compute(self):
        return self.value / self.iter_counter

    def attach(self, engine, name):
        if self.log_interval is not None:
            engine.add_event_handler(Events.ITERATION_STARTED, self.started)  # Reset if needed
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)  # Update
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name) # Compute
        else:
            engine.add_event_handler(Events.EPOCH_STARTED, self.started) # Reset
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)  # Update
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)  # Compute
