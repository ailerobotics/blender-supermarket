"""Parameters for the approved controlled-static-scene navigation mapper.

No node, GPU, dependency install or process lifecycle is started here. Host owns
the /restock_nav mapper; /restock_sim parked manipulation remains independent.
"""
import math


def navigation_mapper_parameters(*,camera_count,transport_height,frame='map',voxel_size=.05):
    if (type(camera_count) is not int or not 1<=camera_count<=4 or frame!='map'
            or not math.isfinite(transport_height) or transport_height<1.65
            or not math.isfinite(voxel_size) or not 0<voxel_size<=.05):
        raise ValueError('1-4 cameras, stable map frame and measured complete transport height required')
    return {'use_sim_time':True,'global_frame':frame,'use_tf_transforms':True,
        'mapping_type':'static_tsdf','num_cameras':camera_count,'voxel_size':voxel_size,
        'use_depth':True,'use_color':False,'use_lidar':False,'esdf_mode':'2d',
        'publish_esdf_distance_slice':True,'integrate_depth_rate_hz':20.,
        'update_esdf_rate_hz':10.,'update_mesh_rate_hz':0.,'map_clearing_frame_id':'base_link',
        'map_clearing_radius_m':15.,'static_mapper.tsdf_set_free_distance_on_decayed':False,
        'static_mapper.esdf_slice_min_height':.03,
        'static_mapper.esdf_slice_max_height':transport_height+.05,
        'static_mapper.esdf_slice_height':.03,
        'static_mapper.workspace_bounds_type':'height_bounds',
        'static_mapper.workspace_bounds_min_height_m':.03,
        'static_mapper.workspace_bounds_max_height_m':transport_height+.05}
