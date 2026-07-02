"""
Thin dispatcher: delegates nested cross-validation to eval_crossval.main_crossval.
Supports `eval_name = "video_classification_frozen_crossval"` in the launch scaffold.
"""

from evals.video_classification_frozen.eval_crossval import main_crossval as main
