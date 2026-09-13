from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class TaskExecutionSpec:
    class_id: int
    subcat: str
    task_type: str
    input_mode: str
    reference_source: str
    target_success_count: int = 5


TASK_EXECUTION_SPECS: Dict[Tuple[int, str], TaskExecutionSpec] = {
    (1, "two_image_single_action"): TaskExecutionSpec(
        class_id=1,
        subcat="two_image_single_action",
        task_type="action_inference",
        input_mode="multi_frame_sequence",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/1/meta/qa_data.json",
    ),
    (1, "two_image_double_action"): TaskExecutionSpec(
        class_id=1,
        subcat="two_image_double_action",
        task_type="action_inference",
        input_mode="multi_frame_sequence",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/1/meta/qa_data.json",
    ),
    (1, "multi_image_single_action_chain"): TaskExecutionSpec(
        class_id=1,
        subcat="multi_image_single_action_chain",
        task_type="action_inference",
        input_mode="multi_frame_sequence",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/1/meta/qa_data.json",
    ),
    (2, "default"): TaskExecutionSpec(
        class_id=2,
        subcat="default",
        task_type="multi_image_overlap_localization",
        input_mode="triple_frame_relation",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/2/meta/qa_data.json",
    ),
    (3, "default"): TaskExecutionSpec(
        class_id=3,
        subcat="default",
        task_type="parallax_depth_inference",
        input_mode="dual_frame_depth",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/3/meta/qa_data.json",
    ),
    (4, "start_visible"): TaskExecutionSpec(
        class_id=4,
        subcat="start_visible",
        task_type="movement_sequence_sorting",
        input_mode="four_frame_ordering",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/4/meta/qa_data.json",
    ),
    (4, "start_not_visible"): TaskExecutionSpec(
        class_id=4,
        subcat="start_not_visible",
        task_type="movement_sequence_sorting",
        input_mode="four_frame_ordering",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/4/meta/qa_data.json",
    ),
    (5, "translation"): TaskExecutionSpec(
        class_id=5,
        subcat="translation",
        task_type="movement_degree_comparison",
        input_mode="triple_frame_comparison",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/5/meta/qa_data.json",
    ),
    (5, "rotation"): TaskExecutionSpec(
        class_id=5,
        subcat="rotation",
        task_type="movement_degree_comparison",
        input_mode="triple_frame_comparison",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/5/meta/qa_data.json",
    ),
    (6, "rotate"): TaskExecutionSpec(
        class_id=6,
        subcat="rotate",
        task_type="object_state_attribute_changes",
        input_mode="dual_frame_state_change",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/6/meta/qa_data.json",
    ),
    (6, "open_close"): TaskExecutionSpec(
        class_id=6,
        subcat="open_close",
        task_type="object_state_attribute_changes",
        input_mode="dual_frame_state_change",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/6/meta/qa_data.json",
    ),
    (6, "toggle"): TaskExecutionSpec(
        class_id=6,
        subcat="toggle",
        task_type="object_state_attribute_changes",
        input_mode="dual_frame_state_change",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/6/meta/qa_data.json",
    ),
    (6, "remove"): TaskExecutionSpec(
        class_id=6,
        subcat="remove",
        task_type="object_state_attribute_changes",
        input_mode="dual_frame_state_change",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/6/meta/qa_data.json",
    ),
    (7, "default"): TaskExecutionSpec(
        class_id=7,
        subcat="default",
        task_type="object_position_swapping",
        input_mode="dual_frame_swap",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/7/meta/qa_data.json",
    ),
    (8, "distance_change"): TaskExecutionSpec(
        class_id=8,
        subcat="distance_change",
        task_type="dynamic_movement_occlusion",
        input_mode="dual_frame_motion_change",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/8/meta/qa_data.json",
    ),
    (8, "occlusion_change"): TaskExecutionSpec(
        class_id=8,
        subcat="occlusion_change",
        task_type="dynamic_movement_occlusion",
        input_mode="dual_frame_motion_change",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/8/meta/qa_data.json",
    ),
    (9, "default"): TaskExecutionSpec(
        class_id=9,
        subcat="default",
        task_type="imagined_perspective_taking",
        input_mode="single_frame_reasoning",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/9/meta/qa_data.json",
    ),
    (10, "single_visibility"): TaskExecutionSpec(
        class_id=10,
        subcat="single_visibility",
        task_type="imagined_movement_consequence",
        input_mode="single_frame_reasoning",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/10/meta/qa_data.json",
    ),
    (10, "two_frame_direction"): TaskExecutionSpec(
        class_id=10,
        subcat="two_frame_direction",
        task_type="imagined_movement_consequence",
        input_mode="single_frame_reasoning",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/10/meta/qa_data.json",
    ),
    (10, "single_direction"): TaskExecutionSpec(
        class_id=10,
        subcat="single_direction",
        task_type="imagined_movement_consequence",
        input_mode="single_frame_reasoning",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/10/meta/qa_data.json",
    ),
    (10, "trend_disappear"): TaskExecutionSpec(
        class_id=10,
        subcat="trend_disappear",
        task_type="imagined_movement_consequence",
        input_mode="single_frame_reasoning",
        reference_source="/path/to/workspace/SCENE/HSSD/local_tasks/10/meta/qa_data.json",
    ),
}


def spec_for(class_id: int, subcat: str) -> TaskExecutionSpec:
    return TASK_EXECUTION_SPECS[(int(class_id), str(subcat))]


def all_subcategory_keys() -> List[str]:
    return [f"{spec.class_id}_{spec.subcat}" for spec in TASK_EXECUTION_SPECS.values()]
