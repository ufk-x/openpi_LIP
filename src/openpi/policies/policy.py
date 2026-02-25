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
            # NOTE:
            # We intentionally do NOT use nnx_utils.module_jit here.
            # module_jit freezes module state at wrap time, which would make
            # runtime-updated inference attributes (e.g. rtc_guidance,
            # replan_steps, guidance_* set per request) stale during sampling.
            # Calling the bound method directly keeps per-request config effective.
            # self._sample_actions = nnx_utils.module_jit(model.sample_actions) # 因为sample_actions里可能会用到rtc_guidance等推理时才会设置的类成员，所以不能jit封装函数，假如改成形参传入，那么就可以使用封装后的函数了
            self._sample_actions = model.sample_actions
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        rtc = obs.get("rtc")
        if hasattr(self._model, "rtc_guidance") and rtc is not None:
            self._model.rtc_guidance = bool(rtc)

        replan_steps = obs.get("replan_steps")
        if hasattr(self._model, "replan_steps") and replan_steps is not None:
            replan_steps = int(replan_steps)
            self._model.replan_steps = replan_steps
            if hasattr(self._model, "guidance_prefix_attention_horizon") and hasattr(self._model, "action_horizon"):
                self._model.guidance_prefix_attention_horizon = self._model.action_horizon - replan_steps

        delay_steps = obs.get("delay_steps")
        if hasattr(self._model, "guidance_inference_delay") and delay_steps is not None:
            self._model.guidance_inference_delay = int(delay_steps)

        guidance_schedule = obs.get("guidance_prefix_attention_schedule")
        if hasattr(self._model, "guidance_prefix_attention_schedule") and guidance_schedule is not None:
            self._model.guidance_prefix_attention_schedule = str(guidance_schedule)

        guidance_max_weight = obs.get("guidance_max_weight")
        if hasattr(self._model, "guidance_max_weight") and guidance_max_weight is not None:
            self._model.guidance_max_weight = float(guidance_max_weight)

        if hasattr(self._model, "rtc_guidance_chunk"):
            rtc_guidance_chunk = obs.get("rtc_guidance_chunk")
            if rtc_guidance_chunk is None:
                self._model.rtc_guidance_chunk = None
            else:
                rtc_guidance_chunk = np.asarray(rtc_guidance_chunk)
                if rtc_guidance_chunk.ndim == 2:
                    rtc_guidance_chunk = rtc_guidance_chunk[None, ...]
                model_action_dim = getattr(self._model, "action_dim", None)
                if model_action_dim is not None and rtc_guidance_chunk.shape[-1] != model_action_dim:
                    if rtc_guidance_chunk.shape[-1] < model_action_dim:
                        pad_width = model_action_dim - rtc_guidance_chunk.shape[-1]
                        rtc_guidance_chunk = np.pad(
                            rtc_guidance_chunk,
                            ((0, 0), (0, 0), (0, pad_width)),
                            mode="constant",
                        )
                    else:
                        rtc_guidance_chunk = rtc_guidance_chunk[..., :model_action_dim]
                self._model.rtc_guidance_chunk = jnp.asarray(rtc_guidance_chunk)

        obs = {
            k: v
            for k, v in obs.items()
            if k
            not in {
                "rtc",
                "replan_steps",
                "delay_steps",
                "guidance_prefix_attention_schedule",
                "guidance_max_weight",
                "rtc_guidance_chunk",
            }
        }

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
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
