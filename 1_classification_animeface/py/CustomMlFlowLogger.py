from pytorch_lightning.loggers.logger import Logger
from lightning_fabric.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class CustomMlFlowLogger(Logger):
    def __init__(self, writer):
        super(CustomMlFlowLogger, self).__init__()
        self.writer = writer

    @property
    def name(self):
        return "CustomMlFlowLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for key, value in metrics.items():
            self.writer.log_metric(key, value, step=step)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        self.writer.set_terminated()
