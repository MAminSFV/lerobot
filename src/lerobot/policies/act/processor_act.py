#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
    ]
    
    # Add language processing steps if enabled
    if config.use_language_conditioning:
        input_steps.append(TaskIndexToTaskProcessorStep())
        input_steps.append(
            TokenizerProcessorStep(
                tokenizer_name=config.language_encoder,
                max_length=20,  # Match max_text_len in modeling_act.py
                padding="max_length",
                padding_side="right",
                truncation=True,
            )
        )
    
    input_steps.extend([
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ])
    
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@dataclass
@ProcessorStepRegistry.register(name="task_index_to_task_processor")
class TaskIndexToTaskProcessorStep(ComplementaryDataProcessorStep):
    """
    A processor step that converts task_index to task string using Libero benchmark.
    
    If `task` is already present (e.g., at inference time when operator provides it),
    this step does nothing. Otherwise, it looks up the task description from the
    Libero benchmark using the task_index.
    """
    
    _task_descriptions: list[str] | None = None
    
    def __post_init__(self):
        """Build the task descriptions list from Libero benchmark."""
        if TaskIndexToTaskProcessorStep._task_descriptions is None:
            try:
                from libero.libero import benchmark
                
                task_descriptions = []
                benchmark_dict = benchmark.get_benchmark_dict()
                # Order must match the global indices in the HuggingFaceVLA/libero dataset
                suite_order = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_90', 'libero_10']
                
                for suite_name in suite_order:
                    if suite_name in benchmark_dict:
                        suite = benchmark_dict[suite_name]()
                        for i in range(suite.n_tasks):
                            task_descriptions.append(suite.tasks[i].language)
                
                TaskIndexToTaskProcessorStep._task_descriptions = task_descriptions
            except ImportError:
                # If libero is not available, we can't do the mapping
                TaskIndexToTaskProcessorStep._task_descriptions = []
    
    def complementary_data(self, complementary_data: dict) -> dict:
        """
        Convert task_index to task string if task is not already present.
        
        Args:
            complementary_data: The complementary data dictionary.
            
        Returns:
            Updated complementary data with 'task' field populated.
        """
        # If task is already present, do nothing
        if "task" in complementary_data and complementary_data["task"] is not None:
            return complementary_data
        
        # If task_index is present, look up the task description
        if "task_index" in complementary_data:
            task_index = complementary_data["task_index"]
            
            # Handle tensor or list
            if hasattr(task_index, 'tolist'):
                indices = task_index.tolist()
            elif isinstance(task_index, list):
                indices = task_index
            else:
                indices = [task_index]
            
            # Handle single index vs list
            if isinstance(indices, int):
                indices = [indices]
            
            # Look up task descriptions
            if self._task_descriptions:
                tasks = [self._task_descriptions[i] for i in indices]
                # Return single string if single item, else list
                complementary_data = dict(complementary_data)
                complementary_data["task"] = tasks[0] if len(tasks) == 1 else tasks
        
        return complementary_data
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Returns the input features unchanged."""
        return features
