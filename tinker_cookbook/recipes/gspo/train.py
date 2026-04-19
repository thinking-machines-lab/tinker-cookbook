"""GSPO on GSM8K - see loss.py for the core algorithm."""

import logging
import time

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.gspo.loss import DEFAULT_CLIP_HIGH, DEFAULT_CLIP_LOW, make_gspo_loss
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
