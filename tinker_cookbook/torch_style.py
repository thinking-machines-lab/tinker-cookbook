# pyright: reportAttributeAccessIssue=false
"""
This module provides a torch-style interface for tinker, where you forward-pass your loss and do loss.backward().
At the cost of an extra forward pass, this lets you use any custom loss function -- even one that operates on multiple sequences.
"""

import logging
from typing import List, Tuple

import tinker
import torch
from tinker import types

logger = logging.getLogger(__name__)


class PlaceholderFunction(torch.autograd.Function):
    """
    No-op on forward pass, but will call forward_backward on the backward pass.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        client: tinker.TrainingClient,
        data: List[types.Datum],
        *output_tensor_list: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # Store context for backward
        ctx.client = client
        ctx.data = data

        # Save input shapes for backward
        ctx.input_shapes = [tensor.shape for tensor in output_tensor_list]
        ctx.save_for_backward(*output_tensor_list)

        # Return the tensors as is
        return output_tensor_list

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *grad_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor | None, ...]:
        # Check if shapes match
        if len(grad_outputs) != len(ctx.data):
            raise ValueError(f"Expected {len(ctx.data)} gradients, got {len(grad_outputs)}")

        # Call forward_backward with the linear loss function
        logger.info("Calling forward_backward with linear loss function")
        linear_loss_data = [
            types.Datum(
                model_input=ctx.data[i].model_input,
                loss_fn_inputs={
                    "target_tokens": ctx.data[i].loss_fn_inputs["target_tokens"],
                    "weights": types.TensorData(
                        data=(-grad_outputs[i]).tolist(),
                        dtype="float32",
                        shape=list((-grad_outputs[i]).shape),
                    ),
                },
            )
            for i in range(len(ctx.data))
        ]
        result = ctx.client.forward_backward(linear_loss_data, loss_fn="cross_entropy").result()
        logger.info(f"forward_backward completed with metrics: {result.metrics}")
        # Return None for all inputs to forward except the tensors
        return (None, None) + grad_outputs


def forward_with_autograd(
    client: tinker.TrainingClient,
    data: List[types.Datum],
) -> Tuple[torch.Tensor, ...]:
    """
    Forward pass that returns logprobs with custom autograd behavior.
    Always returns a tuple of tensors, one per datum.

    Args:
        client: The training client to use
        data: A list of Datum objects to process

    Returns:
        A tuple of tensors containing the logprobs for each datum.
    """
    # Do the initial forward pass with cross entropy loss
    outputs = client.forward(data, loss_fn="cross_entropy").result().loss_fn_outputs

    # Extract logprobs from outputs and ensure they require gradients
    logprobs_list: List[torch.Tensor] = []
    for out in outputs:
        logprob = torch.tensor(out["logprobs"].data).clone().detach().requires_grad_(True)
        logprobs_list.append(logprob)

    # Apply the custom autograd function
    result: Tuple[torch.Tensor, ...] = PlaceholderFunction.apply(client, data, *logprobs_list)  # type: ignore

    # Always return the tuple of tensors
    return result


def forward(
    client: tinker.TrainingClient,
    data: List[types.Datum],
) -> Tuple[torch.Tensor, ...]:
    """
    Simple forward pass that returns logprobs without any autograd or gradient handling.
    Always returns a tuple of tensors, one per datum.

    Args:
        client: The training client to use
        data: A list of Datum objects to process

    Returns:
        A tuple of tensors containing the logprobs for each datum.
    """
    # Do the forward pass with cross entropy loss
    outputs = client.forward(data, loss_fn="cross_entropy").result().loss_fn_outputs

    # Extract logprobs from outputs and convert to tensors
    logprobs_list: List[torch.Tensor] = []
    for out in outputs:
        logprob = torch.tensor(out["logprobs"].data)
        logprobs_list.append(logprob)

    # Return the tuple of tensors
    return tuple(logprobs_list)
