import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


def calibration(probability, truth):
    p, y = probability.flatten(), truth.flatten()
    result = {"brier": float(np.mean((p - y) ** 2))}
    ece = 0.
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        selected = (p >= low) & (p < high if high < 1 else p <= high)
        if selected.any():
            ece += np.mean(selected) * abs(p[selected].mean() - y[selected].mean())
    result["ece"] = float(ece)
    return result


def measure(prediction, data):
    truth = np.asarray(data["y"])
    result = {}
    if "probability" in prediction:
        p = prediction["probability"]
        if p.ndim == 2 and truth.ndim == 1:
            labels = truth.astype(int)
            known = (labels >= 0) & (labels < p.shape[1])
            if known.any():
                chosen = p[known].argmax(1)
                result.update(accuracy=float(np.mean(chosen == labels[known])),
                              balanced_accuracy=float(balanced_accuracy_score(labels[known], chosen)),
                              macro_f1=float(f1_score(labels[known], chosen, average="macro", zero_division=0)),
                              nll=float(-np.log(np.clip(p[known, labels[known]], 1e-10, 1)).mean()))
                result.update(calibration(p[known].max(1), chosen == labels[known]))
            if p.shape[1] == 2 and len(np.unique(labels[known])) == 2:
                result.update(auroc=float(roc_auc_score(labels[known], p[known, 1])),
                              auprc=float(average_precision_score(labels[known], p[known, 1])))
            if (~known).any() and known.any():
                unknown = prediction.get("unknown_score", 1 - p.max(1))
                result["unknown_auroc"] = float(roc_auc_score(~known, unknown))
                thresholds = np.r_[-np.inf, np.sort(np.unique(unknown)), np.inf]
                fpr, ccr = [], []
                for threshold in thresholds:
                    accepted = unknown <= threshold
                    fpr.append(float(accepted[~known].mean()))
                    ccr.append(float(np.mean(accepted[known] & (p[known].argmax(1) == labels[known]))))
                result["oscr"] = float(np.trapz(ccr, fpr))
        elif p.shape == truth.shape:
            mask, target = p >= .5, truth >= .5
            tp, fp, fn = np.sum(mask & target), np.sum(mask & ~target), np.sum(~mask & target)
            result.update(dice=float((2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)),
                          iou=float((tp + 1e-8) / (tp + fp + fn + 1e-8)),
                          precision=float(tp / max(tp + fp, 1)),
                          recall=float(tp / max(tp + fn, 1)), mae=float(np.mean(abs(p - truth))))
            result.update(calibration(p, target))
            distances, bf = [], []
            for a, b in zip(mask.reshape(-1, *mask.shape[-2:]), target.reshape(-1, *target.shape[-2:])):
                ea, eb = a ^ binary_erosion(a), b ^ binary_erosion(b)
                if ea.any() and eb.any():
                    da, db = distance_transform_edt(~eb)[ea], distance_transform_edt(~ea)[eb]
                    distances.extend(np.r_[da, db].tolist())
                    precision, recall = np.mean(da <= 2), np.mean(db <= 2)
                    bf.append(2 * precision * recall / max(precision + recall, 1e-10))
                elif ea.any() or eb.any():
                    distances.append(float(np.hypot(*a.shape)))
                    bf.append(0.)
            if distances:
                result.update(hd95=float(np.quantile(distances, .95)), assd=float(np.mean(distances)), boundary_f1=float(np.mean(bf)))
    if "prediction" in prediction and prediction["prediction"].shape == truth.shape:
        value = prediction["prediction"]
        error = value - truth
        result.update(rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(abs(error))), bias=float(error.mean()))
        if truth.ndim == 4:
            mse = np.mean(error ** 2, axis=tuple(range(1, truth.ndim)))
            result["psnr"] = float(np.mean(-10 * np.log10(np.maximum(mse, 1e-12))))
            mx, my = gaussian_filter(value, sigma=(0, 0, 1.5, 1.5)), gaussian_filter(truth, sigma=(0, 0, 1.5, 1.5))
            vx = gaussian_filter(value ** 2, sigma=(0, 0, 1.5, 1.5)) - mx ** 2
            vy = gaussian_filter(truth ** 2, sigma=(0, 0, 1.5, 1.5)) - my ** 2
            covariance = gaussian_filter(value * truth, sigma=(0, 0, 1.5, 1.5)) - mx * my
            result["ssim"] = float(np.mean((2 * mx * my + .01 ** 2) * (2 * covariance + .03 ** 2) /
                ((mx ** 2 + my ** 2 + .01 ** 2) * (vx + vy + .03 ** 2))))
    if {"lower", "upper"} <= prediction.keys() and prediction["lower"].shape == truth.shape:
        lo, hi = prediction["lower"], prediction["upper"]
        result.update(coverage=float(np.mean((truth >= lo) & (truth <= hi))), interval_width=float(np.mean(hi-lo)) if np.isfinite(hi-lo).all() else None)
    group_key = "condition" if "condition" in data else "client" if "client" in data else "group"
    if group_key in data and "probability" in prediction and truth.ndim == 1:
        scores = []
        for group in np.unique(data[group_key]):
            selected = data[group_key] == group
            labels = truth[selected]
            if np.all((labels >= 0) & (labels < prediction["probability"].shape[1])):
                scores.append(f1_score(labels, prediction["probability"][selected].argmax(1), average="macro", zero_division=0))
        if scores:
            result["worst_group_f1"] = float(min(scores))
    return result

