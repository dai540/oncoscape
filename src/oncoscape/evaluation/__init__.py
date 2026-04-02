from .metrics import balanced_accuracy, js_divergence, macro_f1, pearson_mean


def evaluate_and_render(*args, **kwargs):
    from .render import evaluate_and_render as _evaluate_and_render

    return _evaluate_and_render(*args, **kwargs)


__all__ = ["evaluate_and_render", "balanced_accuracy", "js_divergence", "macro_f1", "pearson_mean"]
