'''
hjlib-ground-solver -- the ground 'solving' side extracted from monolith
lib_dynamic_hvip/ground/. Estimate ground parameters / geometry from SMPL
pillars, top-bottom keypoints, depth maps; recover 3D info from 2D HVIP.

The ground 'use' side (reverse_project / transform / by_param) lives in
hjlib-geometry. See docs/design/README.md.
'''

from hjlib_ground_solver.segment_area.cluster_feature import (
    cluster_pixel_features,
)
from hjlib_ground_solver.segment_area.filter_depth_map_feature import (
    get_pixel_features_of_depth_map,
    filter_vertical_and_horizontal_features,
)
from hjlib_ground_solver.segment_area.get_depth_map_by_ground import (
    get_depth_map_by_ground_tensor,
    get_depth_map_by_ground_np,
)
from hjlib_ground_solver.segment_area.get_ground_by_depth_map import (
    forward_ground_parameters_to_depth_map,
    get_depth_map_loss,
    get_ground_main_area_by_depth_map,
    get_list_ground_area_by_depth_map,
)

from hjlib_ground_solver.get_ground_geometry.by_pillars import (
    get_ground_by_pillars_on_the_ground,
)
from hjlib_ground_solver.get_ground_geometry.by_points import (
    compute_plane_normal_by_positions,
    compute_plane_parameters_by_positions_hj,
    get_ground_by_points_on_the_ground_lstsq,
    get_ground_by_points_on_the_ground,
)
from hjlib_ground_solver.get_ground_geometry.by_smpl import (
    get_ground_by_smpls_on_the_ground,
)
from hjlib_ground_solver.get_ground_geometry.by_world_space import (
    get_ground_geometry_in_world_space,
)

from hjlib_ground_solver.get_ground_param.by_world_space import (
    get_ground_param_in_world_space,
    get_ground_param_in_world_space_with_extrinsic,
)

from hjlib_ground_solver.hvip.get_3d_info_from_hvip_2d import (
    get_3d_info_from_hvip_2d,
)

from hjlib_ground_solver.estimate_ground.by_kp_rcr.compute_KN_by_vertical_lines import (
    get_KN,
    get_bias_from_2D_ground_normal,
    get_KN_with_filter,
)
from hjlib_ground_solver.estimate_ground.by_kp_rcr.solve_by_top_bot.project_loss import (
    get_projection_loss,
)
from hjlib_ground_solver.estimate_ground.by_kp_rcr.solve_by_top_bot.search_D import (
    uv_to_xyz_via_ground_torch,
    solve_D_search,
)
from hjlib_ground_solver.estimate_ground.by_kp_rcr.solve_by_top_bot.process_solve_by_top_bot_given_K import (
    solve_ground_param_by_top_bottom_given_K,
)
from hjlib_ground_solver.estimate_ground.by_mesh_lower_envelope import (
    Mesh_Lower_Envelope_Candidate,
    Mesh_Lower_Envelope_Summary,
    compute_per_frame_mesh_minimum_height,
    summarize_mesh_lower_envelope,
)
from hjlib_ground_solver.estimate_ground.by_hj_derived_plantar_zmin import (
    HJ_Derived_Plantar_ZMin_Ground_Provenance,
    HJ_Derived_Plantar_ZMin_Ground_Result,
    HJ_Derived_Plantar_ZMin_Ground_Side,
    estimate_hj_derived_plantar_zmin_ground,
)
from hjlib_ground_solver.estimate_ground.by_mesh_lower_envelope_peeling import (
    Mesh_Lower_Envelope_Peel_Proposal,
    Mesh_Lower_Envelope_Peeling_Config,
    Mesh_Lower_Envelope_Peeling_Result,
    Mesh_Lower_Envelope_Peeling_Status,
    peel_separated_mesh_lower_envelope_prefixes,
)
from hjlib_ground_solver.estimate_ground.by_static_foot_humor import (
    Static_Foot_HuMoR_Cluster,
    Static_Foot_HuMoR_Config,
    Static_Foot_HuMoR_Result,
    Static_Foot_HuMoR_Sample,
    Static_Foot_HuMoR_Status,
    estimate_static_foot_humor_baseline,
)
from hjlib_ground_solver.estimate_ground.by_static_foot_plantar_humor import (
    Static_Foot_Plantar_HuMoR_Cluster,
    Static_Foot_Plantar_HuMoR_Config,
    Static_Foot_Plantar_HuMoR_Result,
    Static_Foot_Plantar_HuMoR_Sample,
    Static_Foot_Plantar_HuMoR_Status,
    estimate_static_foot_plantar_humor_baseline,
)
from hjlib_ground_solver.estimate_ground.by_vertex_subset_observation import (
    Vertex_Subset_Observation_Chunk,
    compute_vertex_subset_observation_chunk,
)
from hjlib_ground_solver.estimate_ground.observation_density import (
    Ground_Observation_Density,
    Ground_Observation_KDE_Density,
    compute_ground_observation_density,
    compute_ground_observation_kde_density,
)
from hjlib_ground_solver.estimate_ground.by_vanishing_direction import (
    Vertical_VP_Selection_Ground_Normal_Result,
    Vanishing_Direction_Ground_Normal_Result,
    solve_ground_normal_by_vertical_vp_selection,
    solve_ground_normal_from_vanishing_directions,
)
from hjlib_ground_solver.estimate_ground.by_orthogonal_vanishing_direction import (
    Orthogonal_Consensus_Ground_Normal_Result,
    solve_ground_normal_by_orthogonal_consensus,
    solve_ground_normal_by_role_aware_orthogonal_consensus,
)
from hjlib_ground_solver.estimate_ground.by_person_vertical_lines import (
    Person_Vertical_Direction_Evidence_Result,
    fit_person_vertical_direction_evidence,
)
from hjlib_ground_solver.estimate_ground.by_equal_vertical_lines import (
    Equal_Weight_Vertical_Line_Ground_Normal_Result,
    Source_Weighted_Vertical_Line_Ground_Normal_Result,
    solve_ground_normal_by_equal_weight_vertical_lines,
    solve_ground_normal_by_source_weighted_vertical_lines,
)

__all__ = [
    # segment_area
    'cluster_pixel_features',
    'get_pixel_features_of_depth_map',
    'filter_vertical_and_horizontal_features',
    'get_depth_map_by_ground_tensor',
    'get_depth_map_by_ground_np',
    'forward_ground_parameters_to_depth_map',
    'get_depth_map_loss',
    'get_ground_main_area_by_depth_map',
    'get_list_ground_area_by_depth_map',
    # get_ground_geometry
    'get_ground_by_pillars_on_the_ground',
    'compute_plane_normal_by_positions',
    'compute_plane_parameters_by_positions_hj',
    'get_ground_by_points_on_the_ground_lstsq',
    'get_ground_by_points_on_the_ground',
    'get_ground_by_smpls_on_the_ground',
    'get_ground_geometry_in_world_space',
    # get_ground_param
    'get_ground_param_in_world_space',
    'get_ground_param_in_world_space_with_extrinsic',
    # hvip
    'get_3d_info_from_hvip_2d',
    # estimate_ground
    'HJ_Derived_Plantar_ZMin_Ground_Provenance',
    'HJ_Derived_Plantar_ZMin_Ground_Result',
    'HJ_Derived_Plantar_ZMin_Ground_Side',
    'estimate_hj_derived_plantar_zmin_ground',
    'get_KN',
    'get_bias_from_2D_ground_normal',
    'get_KN_with_filter',
    'get_projection_loss',
    'uv_to_xyz_via_ground_torch',
    'solve_D_search',
    'solve_ground_param_by_top_bottom_given_K',
    'Mesh_Lower_Envelope_Candidate',
    'Mesh_Lower_Envelope_Summary',
    'compute_per_frame_mesh_minimum_height',
    'summarize_mesh_lower_envelope',
    'Mesh_Lower_Envelope_Peel_Proposal',
    'Mesh_Lower_Envelope_Peeling_Config',
    'Mesh_Lower_Envelope_Peeling_Result',
    'Mesh_Lower_Envelope_Peeling_Status',
    'peel_separated_mesh_lower_envelope_prefixes',
    'Static_Foot_HuMoR_Cluster',
    'Static_Foot_HuMoR_Config',
    'Static_Foot_HuMoR_Result',
    'Static_Foot_HuMoR_Sample',
    'Static_Foot_HuMoR_Status',
    'estimate_static_foot_humor_baseline',
    'Static_Foot_Plantar_HuMoR_Cluster',
    'Static_Foot_Plantar_HuMoR_Config',
    'Static_Foot_Plantar_HuMoR_Result',
    'Static_Foot_Plantar_HuMoR_Sample',
    'Static_Foot_Plantar_HuMoR_Status',
    'estimate_static_foot_plantar_humor_baseline',
    'Vertex_Subset_Observation_Chunk',
    'compute_vertex_subset_observation_chunk',
    'Ground_Observation_Density',
    'Ground_Observation_KDE_Density',
    'compute_ground_observation_density',
    'compute_ground_observation_kde_density',
    'Vertical_VP_Selection_Ground_Normal_Result',
    'solve_ground_normal_by_vertical_vp_selection',
    'Vanishing_Direction_Ground_Normal_Result',
    'solve_ground_normal_from_vanishing_directions',
    'Orthogonal_Consensus_Ground_Normal_Result',
    'solve_ground_normal_by_orthogonal_consensus',
    'solve_ground_normal_by_role_aware_orthogonal_consensus',
    'Person_Vertical_Direction_Evidence_Result',
    'fit_person_vertical_direction_evidence',
    'Equal_Weight_Vertical_Line_Ground_Normal_Result',
    'solve_ground_normal_by_equal_weight_vertical_lines',
    'Source_Weighted_Vertical_Line_Ground_Normal_Result',
    'solve_ground_normal_by_source_weighted_vertical_lines',
]
