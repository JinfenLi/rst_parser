import torch
from torch import Tensor

from torchmetrics import Metric

metric_keys = ['span_f1', 'nuc_f1', 'rel_f1']
class RSTMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("span_precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("span_recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("span_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("nuc_precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("nuc_recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("nuc_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rel_precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rel_recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rel_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mean_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):

        assert preds.shape == target.shape
        span_preds = preds[:, :2]
        span_target = target[:, :2]
        nuc_preds = preds[:, :3]
        nuc_target = target[:, :3]
        rel_preds = torch.cat((preds[:, :2], preds[:, [3]]), dim=1)
        rel_target = torch.cat((target[:, :2], target[:, [3]]), dim=1)
        span_precision, span_recall = hit(span_preds, span_target)
        self.span_f1 += 2 * span_precision * span_recall / (span_precision + span_recall) if (span_precision + span_recall) > 0 else 0.0

        nuc_precision, nuc_recall = hit(nuc_preds, nuc_target)
        self.nuc_f1 += 2 * nuc_precision * nuc_recall / (nuc_precision + nuc_recall) if (nuc_precision + nuc_recall) > 0 else 0.0

        rel_precision, rel_recall = hit(rel_preds, rel_target)
        self.rel_f1 += 2 * rel_precision * rel_recall / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0.0

        self.total += 1

    def compute(self):

        return self.span_f1.float()/self.total, self.nuc_f1.float()/self.total, self.rel_f1.float()/self.total


def init_best_metrics():
    return {
        'best_epoch': 0,
        'dev_best_perf': None,
        'test_best_perf': None,
    }



def init_perf_metrics():

    perf_metrics = torch.nn.ModuleDict({
        'rst_metric': RSTMetric()
    })

    return perf_metrics

def hit(preds: Tensor, target: Tensor):
    preds_list = preds.tolist()
    target_list = target.tolist()
    precision_hit_count = [pred for pred in preds_list if pred in target_list]
    recall_hit_count = [target for target in target_list if target in preds_list]
    return len(precision_hit_count)/len(preds) if len(precision_hit_count) > 0 else 0.0, len(recall_hit_count)/len(target) if len(recall_hit_count) > 0 else 0.0


def calc_f1(preds: Tensor, target: Tensor):
    metric_dict = {}
    span_preds = preds[:, :2]
    span_target = target[:, :2]
    nuc_preds = preds[:, :3]
    nuc_target = target[:, :3]
    rel_preds = torch.cat((preds[:, :2], preds[:, [3]]), dim=1)
    rel_target = torch.cat((target[:, :2], target[:, [3]]), dim=1)

    span_precision, span_recall = hit(span_preds, span_target)
    span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall) if (span_precision + span_recall) > 0 else 0.0

    nuc_precision, nuc_recall = hit(nuc_preds, nuc_target)
    nuc_f1 = 2 * nuc_precision * nuc_recall / (nuc_precision + nuc_recall) if (nuc_precision + nuc_recall) > 0 else 0.0

    rel_precision, rel_recall = hit(rel_preds, rel_target)
    rel_f1 = 2 * rel_precision * rel_recall / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0.0

    metric_dict['span_f1'] = torch.tensor(span_f1) * 100
    metric_dict['nuc_f1'] = torch.tensor(nuc_f1) * 100
    metric_dict['rel_f1'] = torch.tensor(rel_f1) * 100
    return metric_dict



def get_step_metrics(preds, targets, metrics):
    # res = {}
    for key, metric_fn in metrics.items():
        for p, t in zip(preds, targets):
            metric_fn(p, t)


def get_epoch_metrics(metrics):
    res = {}
    for key, metric_fn in metrics.items():
        perf = metric_fn.compute()
        for i, metric in enumerate(metric_keys):
            res.update({metric: perf[i] * 100})
        metric_fn.reset()
    return res