from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _squeeze_leading_batch_jax_or_numpy(x: Any) -> Any:
    """Convert a JAX/NumPy leaf to NumPy and remove leading batch dim if present.

    This is safer than x[0, ...] because debug dictionaries may contain scalar
    leaves such as action_prefix_len or Python ints.
    """
    arr = np.asarray(x)
    if arr.ndim > 0 and arr.shape[0] == 1:
        return arr[0, ...]
    return arr


def _squeeze_leading_batch_torch(x: Any) -> Any:
    """Convert a PyTorch/JAX/NumPy leaf to NumPy and remove leading batch dim if present."""
    if hasattr(x, "detach"):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim > 0 and arr.shape[0] == 1:
        return arr[0, ...]
    return arr


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device.
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
                inputs,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions.
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension.
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim).
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        sampled = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time

        debug = None
        if isinstance(sampled, tuple) and len(sampled) == 2:
            sampled_actions, debug = sampled
        else:
            sampled_actions = sampled

        outputs = {
            "state": inputs["state"],
            "actions": sampled_actions,
        }

        if self._is_pytorch_model:
            outputs = jax.tree.map(_squeeze_leading_batch_torch, outputs)
            if debug is not None:
                debug = jax.tree.map(_squeeze_leading_batch_torch, debug)
        else:
            outputs = jax.tree.map(_squeeze_leading_batch_jax_or_numpy, outputs)
            if debug is not None:
                debug = jax.tree.map(_squeeze_leading_batch_jax_or_numpy, debug)

        outputs = self._output_transform(outputs)

        if debug is not None:
            outputs["debug"] = debug

            # Helpful duplication: after output transforms, actions should be the decoded action buffer.
            # This makes the websocket/client logger independent of where decoding happened.
            outputs["debug"]["output_action_buffer"] = np.asarray(outputs["actions"])

        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
