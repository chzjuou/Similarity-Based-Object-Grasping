import os
import math
import time
import threading
import open3d as o3d
import numpy as np
import pickle
import copy
import pymeshlab.pmeshlab as mlab

from collections import Counter
from itertools import chain
from sklearn.decomposition import PCA


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, math.cos(theta), -math.sin(theta)],
                      [0, math.sin(theta), math.cos(theta)]])

def Ry(theta):
    return np.matrix([[math.cos(theta), 0, math.sin(theta)],
                      [0, 1, 0],
                      [-math.sin(theta), 0, math.cos(theta)]])

def Rz(theta):
    return np.matrix([[math.cos(theta), -math.sin(theta), 0],
                      [math.sin(theta), math.cos(theta), 0],
                      [0, 0, 1]])

def global_registration(source, target):
    # Plane detection based point cloud registration
    largest_plane = 0
    source_planes = source.detect_planar_patches()
    for i in range(len(source_planes)):
        area = source_planes[i].extent[0] * source_planes[i].extent[1]
        if area > largest_plane:
            source_plane_index = i
            largest_plane = area
            w0 = source_planes[i].extent[0]
            h0 = source_planes[i].extent[1]
    if largest_plane == 0:
        return None
    source_bb = source_planes[source_plane_index]
    source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.02, source_bb.center)
    source_frame.rotate(source_bb.R)
    source_frame.translate(source_bb.R.dot([0, 0, -0.007]))

    target.paint_uniform_color([0.5, 0.5, 0.5])
    target_planes = target.detect_planar_patches(min_num_points=10)
    most_similar_plane = 999
    target_plane_index = None
    for j in range(len(target_planes)):
        w = target_planes[j].extent[0]
        h = target_planes[j].extent[1]
        if abs(w - w0) + abs(h - h0) < most_similar_plane:
            target_plane_index = j
            most_similar_plane = abs(w - w0) + abs(h - h0)
    if target_plane_index is None:
        return None
    target_bb = target_planes[target_plane_index]
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.02, target_bb.center)
    target_frame.rotate(target_bb.R)
    target_frame.translate(target_bb.R.dot([0, 0, 0.002]))

    # Find nearest plane
    mindistance = 9999
    for angle_y in [0, math.pi]:
        for angle_z in [0, math.pi, math.pi / 2, -math.pi / 2]:
            source_copy = copy.deepcopy(source)
            rotation1 = source_bb.R
            translation1 = source_bb.center
            transform1 = np.c_[rotation1, translation1.T]
            transform1 = np.r_[transform1, np.array([[0, 0, 0, 1]])]
            rotation2 = target_bb.R.dot(Ry(angle_y)).dot(Rz(angle_z))
            translation2 = target_bb.center
            transform2 = np.c_[rotation2, translation2.T]
            transform2 = np.r_[transform2, np.array([[0, 0, 0, 1]])]
            transformation = transform2.dot(np.linalg.inv(transform1))
            source_copy.transform(transformation)
            source_copy.paint_uniform_color([1, 0.706, 0])
            target.paint_uniform_color([0, 0.651, 0.929])
            if sum(source_copy.compute_point_cloud_distance(target)) < mindistance:
                mindistance = sum(source_copy.compute_point_cloud_distance(target))
                best_transform = transformation
    return best_transform

def ransac_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def icp_registration(source, target, voxel_size, transformation):
    distance_threshold = voxel_size * 0.8
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    radius_normal = voxel_size * 2
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-06, relative_rmse=1e-06, max_iteration=1000))
    return result

def downsample_points_grid(points, grid_size):
    if len(points) == 0:
        return points

    # Compute the grid indices for each point
    grid_indices = np.floor(points / grid_size).astype(int)

    # Use a dictionary to group points by their grid cell
    grid_dict = {}
    for idx, grid_coord in enumerate(map(tuple, grid_indices)):
        if grid_coord not in grid_dict:
            grid_dict[grid_coord] = []
        grid_dict[grid_coord].append(points[idx])

    # Choose a representative point from each grid cell (e.g., the centroid)
    downsampled_points = [
        np.mean(cell_points, axis=0) for cell_points in grid_dict.values()
    ]

    return np.array(downsampled_points)

def ss():
    # LLM based semantic matching
    with open('/home/ch/research/NEDO-project-main/downsample_pcd/category.txt', 'r') as f:
        category = f.read().strip()
    if "_" in category:
        if category.find("_(") != -1:
            category = category[:category.find("_(")]
        category = category[category.find("_") + 1:]
    global ss_list
    ss_list = []
    for home, dirs, files in os.walk('../../../../database/mesh'):
        for filename in files:
            if "_" in filename:
                name = filename[:filename.find("_")]
            else:
                name = filename[:-4]
            # The following candidate lists are extracted from LLM results for the categories 'box', 'bottle', and 'can'.
            # For additional categories, please refer to the paper and use the appropriate prompts with LLMs.
            if category == 'box':
                if name in ['box', 'block', 'brick', 'cube']:
                    ss_list.append(filename)
            elif category == 'bottle':
                if name in ['bottle', 'can', 'cup', 'pitcher', 'mug']:
                    ss_list.append(filename)
            elif category == 'can':
                if name in ['can', 'bottle', 'cup', 'pitcher', 'mug', 'marker']:
                    ss_list.append(filename)

def gs():
    # C-FPFH based geometric matching
    pcd_tree = o3d.geometry.KDTreeFlann(source_pcd)
    source_pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source_pcd,
       o3d.geometry.KDTreeSearchParamHybrid(0.01, 11))
    source_down = source_pcd.voxel_down_sample(0.015)
    binlist = []
    idlist = []
    pointlist, normalist = [], []
    for i in range(len(source_down.points)):
        [num, idx, _] = pcd_tree.search_knn_vector_3d(source_down.points[i], 1)
        if idx[0] not in ind:
            continue
        data = source_pcd_fpfh.data[:, i]
        bin1 = np.argsort(data, axis=0)[-1]
        bin2 = np.argsort(data, axis=0)[-2]
        binlist.append(tuple(sorted((bin1, bin2))))
        idlist.append(idx[0])
        pointlist.append(source_down.points[i])
        normalist.append(source_down.normals[i])
    source_down_inliers = o3d.geometry.PointCloud()
    source_down_inliers.points = o3d.utility.Vector3dVector(pointlist)
    np.asarray(source_pcd.colors)[idlist, :] = [1, 0, 0]

    counter = Counter(binlist)
    most_occurrence = counter.most_common(1)[0][0]
    most_occurrence_index = np.where((np.asarray(binlist) == most_occurrence).all(axis=1))[0]
    np.asarray(source_pcd.colors)[[idlist[index] for index in most_occurrence_index], :] = [0, 0, 1]
    pcd_tree_down = o3d.geometry.KDTreeFlann(source_down_inliers)
    point_neighbor = []
    for index in most_occurrence_index:
        [num, kdx, _] = pcd_tree_down.search_radius_vector_3d(source_down_inliers.points[index], 0.02)
        point_neighbor.append(list(kdx[1:]))
    maxlength = 0
    old_point_index = []
    for ii in range(len(most_occurrence_index)):
        if ii in old_point_index:
            continue
        continuous_point_index = [ii]
        length = 1
        while True:
            old_length = length
            count = 0
            for neighbor in chain.from_iterable([point_neighbor[ii] for ii in continuous_point_index]):
                position = 0
                for element in continuous_point_index:
                    position += len(point_neighbor[element])
                    if position > count:
                        break
                source_normal = normalist[most_occurrence_index[element]]
                query_normal = normalist[neighbor]
                angle_between_normals = np.arccos(np.dot(source_normal, query_normal) / \
                                             (np.linalg.norm(source_normal) * np.linalg.norm(query_normal)))
                if (neighbor in most_occurrence_index and angle_between_normals < math.pi / 12 and \
                        np.where(np.asarray(most_occurrence_index) == neighbor)[0][0] not in continuous_point_index):
                    continuous_point_index.append(np.where(np.asarray(most_occurrence_index) == neighbor)[0][0])
                    length += 1
                count += 1
            if old_length == length:
                old_point_index.extend(continuous_point_index)
                break
        if length > maxlength:
            maxlength = length
            most_continuous_points = continuous_point_index

    X = np.asarray(source_pcd.points)[[idlist[most_occurrence_index[index]] for index in most_continuous_points], :]
    pca = PCA(n_components=3)
    pca.fit(X)
    sigma1 = pca.singular_values_[0]
    sigma2 = pca.singular_values_[1]
    sigma3 = pca.singular_values_[2]
    normalized_sigma1 = sigma1 / math.sqrt(sigma1**2 + sigma2**2 + sigma3**2)
    normalized_sigma2 = sigma2 / math.sqrt(sigma1**2 + sigma2**2 + sigma3**2)
    normalized_sigma3 = sigma3 / math.sqrt(sigma1**2 + sigma2**2 + sigma3**2)
    pca_result = [normalized_sigma1, normalized_sigma2, normalized_sigma3]

    # Determine similar candidates based on quantitative similarity (QS) and distributional similarity (DS)
    global gs_list
    gs_list = []
    for home, dirs, files in os.walk('../../../../database/mesh'):
        for filename in files:
            with open(f'QS/{filename[:-4]}.pickle', 'rb') as inputfile:
                counter0 = pickle.load(inputfile)
            with open(f'DS/{filename[:-4]}.pickle', 'rb') as inputfile:
                counter0_cont = pickle.load(inputfile)
            if most_occurrence in list(counter0_cont.keys()):
                pca_list = counter0_cont[most_occurrence]
            else:
                pca_list = []
            if pca_list == []:
                continue
            total = 0
            for element in counter:
                if counter[element] <= counter0[element]:
                    total = total + counter[element]
                else:
                    total = total + counter0[element]
            m1 = total / len(binlist)
            m2 = min([math.dist(pca_result, x) for x in pca_list])
            print(filename, m1, m2)
            if m1 < 0.9 or m2 > 0.1:
                continue
            gs_list.append(filename)

def ds():
    # SOBB based dimensional matching
    source_pcd_copy = copy.deepcopy(source_pcd)
    transform = np.loadtxt('../../../../results/campos.txt')
    source_pcd_copy.transform(transform)
    plane_normal = np.array([0, 0, 1])
    plane_normal_normalized = plane_normal / np.linalg.norm(plane_normal)
    projection_matrix = np.eye(3) - np.outer(plane_normal_normalized, plane_normal_normalized)
    projected_points = np.asarray(source_pcd_copy.points) @ projection_matrix.T
    downsampled_points = downsample_points_grid(projected_points[:, :2], 0.002)
    zeros_column = np.zeros((*downsampled_points.shape[:-1], 1))
    downsampled_points = np.concatenate((downsampled_points, zeros_column), axis=-1)
    pca = PCA(n_components=3)
    pca.fit(downsampled_points)
    sigma1 = pca.singular_values_[0]
    sigma2 = pca.singular_values_[1]
    sigma3 = pca.singular_values_[2]
    singular_values = [sigma1, sigma2, sigma3]
    index1 = np.argsort(singular_values)[-1]
    index2 = np.argsort(singular_values)[-2]
    vec1 = pca.components_[index1]
    vec2 = pca.components_[index2]
    vec3 = plane_normal_normalized
    rotmat = np.column_stack([vec1, vec2, vec3])
    source_pcd_copy_copy = copy.deepcopy(source_pcd_copy)
    source_pcd_copy_copy.rotate(rotmat.T, center=(0, 0, 0))
    aabb_rotated = source_pcd_copy_copy.get_axis_aligned_bounding_box()
    center = aabb_rotated.get_center()
    max_bound = aabb_rotated.get_max_bound()
    min_bound = aabb_rotated.get_min_bound()
    obb_rotated = o3d.geometry.OrientedBoundingBox(center, np.identity(3), max_bound - min_bound)
    obb_rotated.rotate(rotmat, center=(0, 0, 0))
    obj_extent = sorted(obb_rotated.extent)

    global ds_list
    ds_list = []
    for home, dirs, files in os.walk('../../../../database/extent/'):
        for filename in files:
            file_path = os.path.join(home, filename)
            model_extent = np.loadtxt(file_path)
            size_diff = math.dist(obj_extent, model_extent)
            if size_diff < 0.1:
                ds_list.append(filename)

def registration():
    # Global registration + local registration
    global fitness_list, candidate_list, matrix_list
    fitness_list, candidate_list, matrix_list = [], [], []
    for candidate in sim_list:
        target = o3d.io.read_point_cloud(f'../../../../database/mesh/{candidate}')
        source_pcd_copy = copy.deepcopy(source_pcd)
        source_pcd_copy = source_pcd_copy.voxel_down_sample(0.005)
        target = target.voxel_down_sample(0.005)
        init_transform = global_registration(source_pcd_copy, target)
        voxel_size = 0.01
        if init_transform is None:
            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_pcd_copy,
                                                                                                 target,
                                                                                                 voxel_size)
            result_ransac = ransac_registration(source_down, target_down,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size)
            result_icp = icp_registration(source, target, voxel_size, result_ransac.transformation)
        else:
            result_icp = icp_registration(source_pcd_copy, target, voxel_size, init_transform)
        fitness_list.append(result_icp.fitness)
        candidate_list.append(candidate)
        matrix_list.append(result_icp.transformation)
        #draw_registration_result(source_pcd_copy, target, result_icp.transformation)

# Wait for signal
originalTime = os.path.getmtime('../../../../signals/signal3.txt')
while True:
    #break
    if os.path.getmtime('../../../../signals/signal3.txt') > originalTime:
        print("Start similarity matching")
        break

# Smooth point cloud normals
ms = mlab.MeshSet()
ms.load_new_mesh('../../../../results/obj_down.ply')  # object.ply multi_objects/obj_1.ply
ms.compute_normal_for_point_clouds(k=10, smoothiter=5)
ms.save_current_mesh('object.ply', save_vertex_normal=True)
global source_pcd
source_pcd = o3d.io.read_point_cloud('object.ply')
source_pcd.paint_uniform_color([0.5, 0.5, 0.5])
cl, ind = source_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1)

# Multi-level similarity matching
t1 = threading.Thread(target=ss)
t2 = threading.Thread(target=gs)
t3 = threading.Thread(target=ds)
t1.start()
t2.start()
t3.start()
t1.join()
t2.join()
t3.join()
three_sim, two_sim, one_sim, zero_sim = [], [], [], []
for home, dirs, files in os.walk('../../../../database/mesh'):
    for filename in files:
        count = 0
        if filename in ss_list:
            count += 1
        if filename in gs_list:
            count += 1
        if filename in ds_list:
            count += 1
        if count == 3:
            three_sim.append(filename)
        elif count == 2:
            two_sim.append(filename)
        elif count == 1:
            one_sim.append(filename)
        else:
            zero_sim.append(filename)

global sim_list
sim_list = []
if three_sim != []:
    sim_list = three_sim
    if len(three_sim) < 3:
        sim_list = three_sim + two_sim
elif two_sim != []:
    sim_list = two_sim
    if len(two_sim) < 3:
        sim_list = two_sim + one_sim
elif one_sim != []:
    sim_list = one_sim
else:
    sim_list = zero_sim
print('ss_list:', ss_list)
print('gs_list:', gs_list)
print('ds_list:', ds_list)

# Rank candidates based on point cloud registration
t = threading.Thread(target=registration)
t.start()
t.join()
sorted_index = [index for index, num in sorted(enumerate(fitness_list), key=lambda x: x[-1], reverse=True)]
sorted_score = [num for index, num in sorted(enumerate(fitness_list), key=lambda x: x[-1], reverse=True)]
selected_model, tf_matrix = [], []
for index in sorted_index:
    selected_model.append(candidate_list[index])
    tf_matrix.append(matrix_list[index])
if os.path.exists('../../../../results/candidates.txt'):
    os.remove('../../../../results/candidates.txt')
for candidates in selected_model:
    with open('../../../../results/candidates.txt', 'a') as f:
        f.write(str(candidates)[:-4] + '\n')
np.save('../../../../results/tf_matrix.npy', tf_matrix)
print('Model candidates:', selected_model)
