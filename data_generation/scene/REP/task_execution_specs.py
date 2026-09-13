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
    (1, "two_image_single_action"): TaskExecutionSpec(1, "two_image_single_action", "action_inference", "multi_frame_sequence", "/path/to/workspace/SCENE/REP/local_tasks/1/meta/qa_data.json"),
    (1, "two_image_double_action"): TaskExecutionSpec(1, "two_image_double_action", "action_inference", "multi_frame_sequence", "/path/to/workspace/SCENE/REP/local_tasks/1/meta/qa_data.json"),
    (1, "multi_image_single_action_chain"): TaskExecutionSpec(1, "multi_image_single_action_chain", "action_inference", "multi_frame_sequence", "/path/to/workspace/SCENE/REP/local_tasks/1/meta/qa_data.json"),
    (2, "default"): TaskExecutionSpec(2, "default", "multi_image_overlap_localization", "triple_frame_relation", "/path/to/workspace/SCENE/REP/local_tasks/2/meta/qa_data.json"),
    (3, "default"): TaskExecutionSpec(3, "default", "parallax_depth_inference", "dual_frame_depth", "/path/to/workspace/SCENE/REP/local_tasks/3/meta/qa_data.json"),
    (4, "start_visible"): TaskExecutionSpec(4, "start_visible", "movement_sequence_sorting", "four_frame_ordering", "/path/to/workspace/SCENE/REP/local_tasks/4/meta/qa_data.json"),
    (4, "start_not_visible"): TaskExecutionSpec(4, "start_not_visible", "movement_sequence_sorting", "four_frame_ordering", "/path/to/workspace/SCENE/REP/local_tasks/4/meta/qa_data.json"),
    (5, "translation"): TaskExecutionSpec(5, "translation", "movement_degree_comparison", "triple_frame_comparison", "/path/to/workspace/SCENE/REP/local_tasks/5/meta/qa_data.json"),
    (5, "rotation"): TaskExecutionSpec(5, "rotation", "movement_degree_comparison", "triple_frame_comparison", "/path/to/workspace/SCENE/REP/local_tasks/5/meta/qa_data.json"),
    (6, "rotate"): TaskExecutionSpec(6, "rotate", "object_state_attribute_changes", "dual_frame_state_change", "/path/to/workspace/SCENE/REP/local_tasks/6/meta/qa_data.json"),
    (6, "open_close"): TaskExecutionSpec(6, "open_close", "object_state_attribute_changes", "dual_frame_state_change", "/path/to/workspace/SCENE/REP/local_tasks/6/meta/qa_data.json"),
    (6, "toggle"): TaskExecutionSpec(6, "toggle", "object_state_attribute_changes", "dual_frame_state_change", "/path/to/workspace/SCENE/REP/local_tasks/6/meta/qa_data.json"),
    (6, "remove"): TaskExecutionSpec(6, "remove", "object_state_attribute_changes", "dual_frame_state_change", "/path/to/workspace/SCENE/REP/local_tasks/6/meta/qa_data.json"),
    (7, "default"): TaskExecutionSpec(7, "default", "object_position_swapping", "dual_frame_swap", "/path/to/workspace/SCENE/REP/local_tasks/7/meta/qa_data.json"),
    (8, "distance_change"): TaskExecutionSpec(8, "distance_change", "dynamic_movement_occlusion", "dual_frame_motion_change", "/path/to/workspace/SCENE/REP/local_tasks/8/meta/qa_data.json"),
    (8, "occlusion_change"): TaskExecutionSpec(8, "occlusion_change", "dynamic_movement_occlusion", "dual_frame_motion_change", "/path/to/workspace/SCENE/REP/local_tasks/8/meta/qa_data.json"),
    (9, "default"): TaskExecutionSpec(9, "default", "imagined_perspective_taking", "single_frame_reasoning", "/path/to/workspace/SCENE/REP/local_tasks/9/meta/qa_data.json"),
    (10, "single_visibility"): TaskExecutionSpec(10, "single_visibility", "imagined_movement_consequence", "single_frame_reasoning", "/path/to/workspace/SCENE/REP/local_tasks/10/meta/qa_data.json"),
    (10, "two_frame_direction"): TaskExecutionSpec(10, "two_frame_direction", "imagined_movement_consequence", "single_frame_reasoning", "/path/to/workspace/SCENE/REP/local_tasks/10/meta/qa_data.json"),
    (10, "single_direction"): TaskExecutionSpec(10, "single_direction", "imagined_movement_consequence", "single_frame_reasoning", "/path/to/workspace/SCENE/REP/local_tasks/10/meta/qa_data.json"),
    (10, "trend_disappear"): TaskExecutionSpec(10, "trend_disappear", "imagined_movement_consequence", "single_frame_reasoning", "/path/to/workspace/SCENE/REP/local_tasks/10/meta/qa_data.json"),
}


def spec_for(class_id: int, subcat: str) -> TaskExecutionSpec:
    return TASK_EXECUTION_SPECS[(int(class_id), str(subcat))]


def all_subcategory_keys() -> List[str]:
    return [f"{spec.class_id}_{spec.subcat}" for spec in TASK_EXECUTION_SPECS.values()]
