import torch
from .metric import compute_metrics
from .utils import nested_numpify


def build_validator(config):
    if config.dataset.get('name', None) == 'glue':
        return GlueValidator(config)
    else:
        raise NotImplementedError("Dataset name not provided/supported")


class BaseValidator:
    def __init__(self, config):
        self.config = config
    
    def init_metrics(self):
        raise NotImplementedError("Not implemented")
    
    def update_metrics(self, preds, labels):
        raise NotImplementedError("Not implemented")
    
    def get_desc(self):
        raise NotImplementedError("Not implemented")
    
    def get_results(self):
        return NotImplementedError("Not implemented")

    def print_metrics(self, metrics):
        """Prints the evaluation metrics in a properly aligned format."""
        # Print the header
        print(self.get_desc())
        format_str = "%22s %11d" + " %11.4f" * len(self.metric_names)
        task_name = self.config.task.get('name', None)
        print(format_str % (
                    task_name,
                    self.num_cnts,
                    *metrics.values())
              )


class GlueValidator(BaseValidator):
    def __init__(self, config):
        self.config = config
        self.metric_names = []
        self.init_metrics()

    def init_metrics(self):
        self.num_cnts = 0
        self.metrics = {}
    
    def get_desc(self):
        """Returns description of evaluation metrics in string format."""
        format_str = "%22s %11s" + " %11s" * len(self.metric_names)
        return format_str % (
                    "Task",
                    "Instances",
                    *self.metric_names
                )
        
    def get_metrics(self, preds, labels):
        preds = nested_numpify(preds)
        labels = nested_numpify(labels)
        dataset = self.config.dataset.get('name', None)
        if dataset == 'glue':
            task = self.config.task.get('name', None)
            dtype = labels.dtype
            if task is not None:
                is_regression = task == "stsb"
            elif dtype is not None:
                is_regression = dtype in ["float32", "float64"]
            else:
                raise ValueError("dtype or task_name must be provided")
            return compute_metrics(dataset, task, preds, labels, dtype, is_regression=is_regression)
        else:
            raise NotImplementedError("Dataset name not provided/supported")
    
    def update_metrics(self, results):
        '''Update and accum the metrics with the new predictions and labels.'''
        self.num_cnts += 1
        if not self.metric_names:
            self.metric_names = list(results.keys())

        for k, v in results.items():
            if k not in self.metrics:
                self.metrics[k] = 0
            if isinstance(v, torch.Tensor):
                self.metrics[k] += v.item()
            else:
                self.metrics[k] += v
                
    def get_results(self):
        metrics_results = {}
        for k, v in self.metrics.items():
            metrics_results[k] = v / self.num_cnts
        self.print_metrics(metrics_results)
        return metrics_results
        