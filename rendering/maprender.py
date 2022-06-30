import numpy as np
import open3d as o3d
import cv2


class GridMapRenderer:
	def __init__(self, map):
		self.map = map
		self.ctgr_height = [0, 0, 9, 6, 3, 6]
		self.mesh_verts = np.zeros((1,3), dtype=int)
		self.mesh_trias = np.zeros((1,3), dtype=int)
		self.ctgr_vert_tri = {}
		self.create_mesh(self.map)
		self.ctgr_colors = [[0, 0, 0], [1, 1, 1], [0.6, 0.6, 0.6], [0.6, 0.4, 0], [1, 1, 0], [0.058, 0.878, 1]]
		self.draw_2d()
		self.draw_3d(self.ctgr_vert_tri)
		# x, c = self.create_cube_mesh(map)
		# self.grids_topleft = x
		# self.grids_ctgr = c

	def create_mesh(self, map):
		grids_topleft, grid_ctgrs = self.get_grid_coordinates(map)
		for grid_tl, grid_ctgr in zip(grids_topleft, grid_ctgrs):
			cube_verts, cube_trias = self.create_cube_mesh(grid_tl, self.ctgr_height[grid_ctgr])
			cube_verts, cube_trias = self.update_mesh(cube_verts, cube_trias)
			self.ctgr_vert_tri[grid_ctgr] = [cube_verts, cube_trias]

	def get_grid_coordinates(self, map):
		with open(map, 'r') as f:
			grid_map = np.loadtxt(f)
		grid_map = grid_map.astype(int)
		grid_map = np.flip(grid_map)
		grid_map = np.rot90(grid_map)
		category_count = len(np.unique(grid_map))
		coordinate_list = []
		category_list = []
		for i in range(category_count):
			coordinate = np.where(grid_map == i)
			coordinate = np.dstack(coordinate)
			coordinate = np.reshape(coordinate, (-1, 2))
			coordinate_list.append(coordinate)
			category_list.append(i)
		return coordinate_list, category_list

	def create_cube_mesh(self, grid_tl, grid_ctgr):
		verts = self.get_cube_vertices(grid_tl, grid_ctgr)
		tria = self.get_triangles(grid_tl.shape[0])		#numbers of point
		return verts, tria

	def get_cube_vertices(self, grid_tl, height):
		relative_xyz = np.array([[0, 0, 0],
							     [1, 0, 0],
							     [0, 1, 0],
							     [1, 1, 0],
							     [0, 0, height],
							     [1, 0, height],
							     [0, 1, height],
							     [1, 1, height]], dtype=np.int32)
		vertices = []
		grid_tl = np.pad(grid_tl, ((0, 0), (0, 1)), 'constant', constant_values=0)
		grid_tl = np.repeat(grid_tl, 8, axis=0)
		grid_tl = np.reshape(grid_tl, (-1, 8, 3))
		# for rel_xyz in relative_xyz:
		# 	cube_vertex = grid_tl + rel_xyz
		# 	vertices.append(cube_vertex)
		for single_grid in grid_tl:
			single_cube = single_grid + relative_xyz
			vertices.append(single_cube)
		# (N,8,3)
		vertices = np.asarray(vertices)
		# (N*8, 3)
		vertices = np.reshape(vertices, (-1, 3))
		return vertices

	def get_triangles(self, num_tri):
		cube_tris = np.array([[0, 4, 2],  # side
							  [4, 6, 2],
							  [2, 6, 3],
							  [6, 7, 3],
							  [3, 7, 1],
							  [7, 5, 1],
							  [1, 5, 0],
							  [5, 4, 0],
							  # opposite direction
							  [2, 4, 0],
							  [2, 6, 4],
							  [3, 6, 2],
							  [3, 7, 6],
							  [1, 7, 3],
							  [1, 5, 7],
							  [0, 5, 1],
							  [0, 4, 5],
							  # top
							  [4, 5, 6],
							  [6, 5, 7],
							  # bottom
							  [2, 1, 0],
							  [3, 1, 2],
							  ], dtype=np.int32)
		cube_tris = np.tile(cube_tris, (num_tri, 1, 1))
		offset = np.arange(0, num_tri * 8, 8).reshape((num_tri, 1, 1))
		cube_tris += offset
		cube_tris = np.reshape(cube_tris, (-1, 3))
		return cube_tris

	def update_mesh(self, new_verts, new_trias):
		unq_vert, vert_idx = np.unique(new_verts, axis=0, return_index=True)
		unq_tri, tri_idx = np.unique(new_trias, axis=0, return_index=True)
		_, vert_index = np.unique(new_verts, axis=0, return_inverse=True)
		updated_vert = new_verts[np.sort(vert_idx)]
		convert_tri = []
		for triangle in new_trias:
			first, second, third = triangle
			conv1, conv2, conv3 = vert_index[first], vert_index[second], vert_index[third]
			convert_tri.append([conv1, conv2, conv3])
		convert_tri = np.reshape(convert_tri, (-1, 3))
		# print(convert_tri)
		# print(convert_tri.shape)
		# updated_tri = new_trias[np.sort(tri_idx)]
		# # new_verts: (N, 3), new_trias: (N, 3)
		# # mesh_verts: (M, 3), mesh_trias: (M, 3)
		# mesh_verts = self.mesh_verts.reshape(-1, 1, 3)	# (M, 1, 3)
		# new_verts = new_verts.reshape(1, -1, 3)			# (1, N, 3)
		return unq_vert, convert_tri

	def draw_2d(self):
		map_2d = np.flip(self.map.copy(), 0)
		map_2d = np.flip(self.map.copy(), 0)
		x_axis, y_axis = map_2d.shape
		img = np.zeros((x_axis * 10, y_axis * 10, 3), np.uint8) + 255
		for ctgr_idx, category in enumerate(self.categories):
			color = self.ctgr_colors[ctgr_idx].copy()
			color = [color[i] * 255 for i in range(len(color))]
			color[0], color[2] = color[2], color[0]
			if ctgr_idx == 0:
				continue
			y, x = np.where(map_2d == ctgr_idx)
			for x_, y_ in zip(x, y):
				img = cv2.rectangle(img, (x_ * 10, y_ * 10), (x_ * 10 + 9, y_ * 10 + 9), color, -1)
		cv2.imshow("2D", img)
		cv2.waitKey()

	def draw_3d(self, annotation):
		meshes = []
		for key, value in zip(annotation.keys(), annotation.values()):
			if key == 0:
				continue
			vert, tri = value
			mesh = self.create_mesh_object(vert, tri, self.ctgr_colors[key])
			meshes.append(mesh)
		o3d.visualization.draw_geometries(meshes)

	def create_mesh_object(self, vertices, triangle, color):
		vert = o3d.utility.Vector3dVector(vertices)
		tri = o3d.utility.Vector3iVector(triangle)
		mesh = o3d.geometry.TriangleMesh(vert, tri)
		mesh.compute_vertex_normals()
		mesh.paint_uniform_color(color)
		return mesh

if __name__ == "__main__":
	map = '/home/cheetah/lee_ws/3D-Map-Renderer/sample_map/sample_18_1.txt'
	Gridmap = GridMapRenderer(map)
