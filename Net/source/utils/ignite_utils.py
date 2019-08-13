from enum import Enum

from ignite.metrics import Metric
from ignite.engine import Events, State


class PeriodicMetric(Metric):

    def __init__(self, output_transform, log_interval):
        self.log_interval = log_interval

        self.value = None

        super().__init__(output_transform)

    def reset(self):
        pass

    def update(self, output):
        self.value = output

    def compute(self):
        return self.value

    def attach(self, engine, name):
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

        engine.add_event_handler(periodic_event, self.iteration_completed)  # Update
        engine.add_event_handler(periodic_event, self.completed, name)  # Compute


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
        self.value += output
        self.iter_counter += 1

    def compute(self):
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
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)


class AveragePeriodicListMetric(AveragePeriodicMetric):

    def __init__(self, output_transform, log_interval=None):
        super().__init__(output_transform, log_interval)

        self.value = None

    def reset(self):
        if self.log_interval is None or self.iter_counter % self.log_interval == 0:
            self.value = None
            self.iter_counter = 0

    def update(self, output):
        if self.value is None:
            self.value = output
        else:
            for i, v in enumerate(output):
                self.value[i] += v

        self.iter_counter += 1

    def compute(self):
        for i in range(len(self.value)):
            self.value[i] /= self.iter_counter

        return self.value


class CollectMetric(Metric):

    def __init__(self, output_transform):
        self.outputs = []

        super().__init__(output_transform)

    def reset(self):
        self.outputs = []

    def update(self, output):
        self.outputs += [output]

    def compute(self):
        return self.outputs

    def attach(self, engine, name):
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)  # Reset
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)  # Update
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)  # Compute

