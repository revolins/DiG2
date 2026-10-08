import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import KLDivLoss
from torchmetrics import Metric


class TrainAbstractMetricsDiscrete(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None





class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, weight: Tensor = None) -> None:
        """Update state with predictions and targets.
        preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
        target: Ground truth values     (bs * n, d) or (bs * n * n, d)."""
        target = torch.argmax(target, dim=-1)
        if weight is not None:
            output = F.cross_entropy(
                preds,
                target,
                reduction="none",
                weight=weight,
            )
            # output = (output * weight).sum()
        else:
            output = F.cross_entropy(
                preds,
                target,
                reduction="sum",
                weight=None,
            )
        # output = F.cross_entropy(preds, target, reduction="sum")
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class KLDMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_ce", default=torch.tensor(0.0), dist_reduce_fx="sum") # accumulate the kl loss
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum") # accumulate 

    def update(self, preds: Tensor, target: Tensor, weight: Tensor = None) -> None:
        """Update state with predictions and targets.
        preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
        target: Ground truth values     (bs * n, d) or (bs * n * n, d)."""
        # target = torch.argmax(target, dim=-1)
        if weight is not None:
            output = KLDivLoss(reduction="none")(
                preds,
                target,
                weight=weight
            )
            # print(output.shape)
            # output = (output * weight).sum()
        else:
            output = KLDivLoss(reduction="none")(
                preds,
                target,
                weight=None
            )

        output[output.isnan()] = 0  # zero-out masked places

        output = output.sum()
        # output = F.cross_entropy(preds, target, reduction="sum")
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples


def compute_ratios(gen_metrics, ref_metrics, metrics_keys):
    print("Computing ratios of metrics: ", metrics_keys)
    if ref_metrics is not None and len(metrics_keys) > 0:
        ratios = {}
        for key in metrics_keys:
            try:
                ref_metric = round(ref_metrics[key], 4)
            except:
                print(key, "not found")
                continue
            if ref_metric != 0.0:
                ratios[key + "_ratio"] = gen_metrics[key] / ref_metric
            else:
                print(f"WARNING: Reference {key} is 0. Skipping its ratio.")
        if len(ratios) > 0:
            ratios["average_ratio"] = sum(ratios.values()) / len(ratios)
        else:
            ratios["average_ratio"] = -1
            print(f"WARNING: no ratio being saved.")
    else:
        print("WARNING: No reference metrics for ratio computation.")
        ratios = {}

    return ratios
