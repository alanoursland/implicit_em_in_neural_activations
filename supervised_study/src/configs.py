"""Ablation configurations for supervised implicit EM study.

Each config maps directly to SupervisedModel constructor kwargs.
Only volume control (variance + decorrelation) is regularization.
EM dynamics come from NegLogSoftmin shaping supervised gradients,
not from an auxiliary LSE loss.
"""

CONFIGS = {
    # No NegLogSoftmin, no regularization. Standard two-layer MLP.
    "baseline": {
        "use_neg_log_softmin": False,
        "lambda_var": 0.0,
        "lambda_tc": 0.0,
    },
    # NegLogSoftmin calibration only, no volume control.
    "nls_only": {
        "use_neg_log_softmin": True,
        "lambda_var": 0.0,
        "lambda_tc": 0.0,
    },
    # NegLogSoftmin + variance (anti-collapse, no anti-redundancy).
    "nls_var": {
        "use_neg_log_softmin": True,
        "lambda_var": 1.0,
        "lambda_tc": 0.0,
    },
    # NegLogSoftmin + variance + decorrelation. Full implicit EM layer.
    "nls_var_tc": {
        "use_neg_log_softmin": True,
        "lambda_var": 1.0,
        "lambda_tc": 1.0,
    },
    # Volume control without calibration. Tests whether VC alone provides structure.
    "var_tc_only": {
        "use_neg_log_softmin": False,
        "lambda_var": 1.0,
        "lambda_tc": 1.0,
    },
}
