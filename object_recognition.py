import pyrealsense2 as rs
import cv2
import numpy as np
import open3d as o3d
import os
import time

cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe = rs.pipeline()
profile = pipe.start(cfg)
alpha = 1.5 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)

try:
	for k in range(30):
		frames = pipe.wait_for_frames()
	start = time.time()
	frames = pipe.wait_for_frames()
	color_frame = frames.get_color_frame()
	depth_frame = frames.get_depth_frame()
	color_image = np.asanyarray(color_frame.get_data())
	depth_image = depth_frame.get_data()
	cv2.imwrite('detic/input/sin.png', color_image)
	cv2.imwrite('sam/input/sin.png', color_image)

	# Localize target object by background differencing
	obj_img = cv2.imread('detic/input/sin.png')
	bg_img = cv2.imread('results/bg.png')
	diff = cv2.absdiff(obj_img, bg_img)
	mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	mask = cv2.convertScaleAbs(mask, alpha=alpha, beta=beta)
	ret, binary = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	maxarea = 0
	for c in contours:
		area = cv2.contourArea(c)
		if area > maxarea:
			maxarea = area
			cmax = c
	x0, y0, w0, h0 = cv2.boundingRect(cmax)
	cv2.rectangle(mask, (x0, y0), (x0 + w0, y0 + h0), (255, 0, 0), 3)
	np.savetxt('results/obj_bounds.txt', np.asarray([x0, y0, x0 + w0, y0 + h0]))
	cv2.imwrite('results/diff.png', mask)
	with open('signals/signal1.txt', 'w') as f:
		f.write('Start instance segmentation.')

	# Read camera intrinsics
	depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
	depth_min = 0.1
	depth_max = 5.0
	depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
	color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
	depth_to_color_extrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color))
	color_to_depth_extrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.depth))

	# Wait for signal
	originalTime = os.path.getmtime('signals/signal2.txt')
	while True:
		#break
		if os.path.getmtime('signals/signal2.txt') > originalTime:
			print("Finish instance segmentation.")
			break

	# Perform instance segmentation
	with open('results/category.txt', 'r') as f:
		category = f.read().strip()
	if 'None' in category:
		os.system("sh sam.sh")
		min_difference = 9999
		for home, dirs, files in os.walk('sam/output/sin/'):
			for filename in files:
				file_path = os.path.join(home, filename)
				if filename.endswith('png'):
					img = cv2.imread(file_path)
					gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
					maxarea = 0
					for c in contours:
						area = cv2.contourArea(c)
						if area > maxarea:
							maxarea = area
							cmax = c
					x, y, w, h = cv2.boundingRect(cmax)
					if abs(x - x0) + abs(y - y0) + abs(w - w0) + abs(h - h0) < min_difference:
						min_difference = abs(x - x0) + abs(y - y0) + abs(w - w0) + abs(h - h0)
						matched_img = img
		keypoints = np.where((matched_img == [255, 255, 255]).all(axis=2))
		color_points = np.asarray([keypoints[0], keypoints[1]]).T
	else:
		color_points = np.loadtxt('results/keypoints.txt')
		mask_img = color_image.copy()
		mask_img[:, :] = [0, 0, 0]
		for color_point in color_points:
			mask_img[int(color_point[0]), int(color_point[1])] = [255, 255, 255]

	# Extract object point cloud
	colors, points = [], []
	color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
	for color_point in color_points:
		color = color_image_rgb[int(color_point[0])][int(color_point[1])]
		color_normalize = color / 255
		colors.append(color_normalize)
		depth_point = rs.rs2_project_color_pixel_to_depth_pixel(
			depth_image, depth_scale,
			depth_min, depth_max,
			depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, [color_point[1], color_point[0]])
		distance = depth_frame.get_distance(int(depth_point[0]), int(depth_point[1]))
		point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_point, distance)
		points.append(point)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
	pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
	pcd.estimate_normals()
	cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1)
	o3d.io.write_point_cloud("results/obj_ori.ply", cl)
	downsample = cl.voxel_down_sample(0.005)
	downsample, ind = downsample.remove_statistical_outlier(nb_neighbors=500, std_ratio=1)
	o3d.io.write_point_cloud('results/obj_down.ply', downsample)
	# o3d.visualization.draw_geometries([cl])

finally:
	with open('signals/signal3.txt', 'w') as f:
		f.write('Start similarity matching.')
	os.system('cp detic/output/sin.png results/seg.png')
	pipe.stop()
	cv2.destroyAllWindows()

