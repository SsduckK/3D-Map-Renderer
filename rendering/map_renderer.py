import numpy as np
import open3d as o3d
import cv2


class MapRenderer:
    def __init__(self, filename):
        self.USE_OBSTACLE = True
        self.SEMANTIC = True
        self.categories = ["nothing", "floor", "wall", "door", "obstacle", "window"]
        self.ctgr_heights = [0, 0.1, 9, 6, 3, 6]
        # TODO
        self.ctgr_colors = [[0, 0, 0], [1, 1, 1], [0.6, 0.6, 0.6], [0.6, 0.4, 0], [1, 1, 0], [0.058, 0.878, 1]]
        self.map = self.read_map(filename)
        v, t = self.create_mesh_data(self.map)
        self.vert_packs = v
        self.tria_packs = t
        self.draw_mesh_2d()
        self.draw_mesh_3d()

    def read_map(self, filename):
        with open(filename, 'r') as f:
            map = np.loadtxt(f)
        if self.SEMANTIC == False:
            np.place(map, map == 3, 2)
            np.place(map, map == 4, 2)
            np.place(map, map == 5, 2)
        if self.USE_OBSTACLE == False:
            np.place(map, map == 4, 1)
        map = map.astype(int)
        map = np.flip(map, 0)
        return map

    def create_mesh_data(self, map):
        vert_packs = [[] for ctgr in self.categories]
        tria_packs = [[] for ctgr in self.categories]
        for c in range(len(self.categories)):
            grid_y, grid_x = np.nonzero(map == c)
            grid_z = np.zeros_like(grid_y)
            # (N, 3)
            grid_xyz = np.stack([grid_x, grid_y, grid_z], axis=1)
            # (N*8, 3) 8 = num vertices per cube
            vertices = self.get_cube_vertices(grid_xyz, self.ctgr_heights[c])
            vert_packs[c] = vertices
            # (N, 8, 3) 8 = num triangles per cube
            print(grid_xyz.shape[0])
            triangles = self.get_triangles(grid_xyz.shape[0])
            tria_packs[c] = triangles
        return vert_packs, tria_packs

    def get_cube_vertices(self, grid_xyz, height):
        relative_xyz = np.array([[0, 0, 0],
                                 [1, 0, 0],
                                 [0, 1, 0],
                                 [1, 1, 0],
                                 [0, 0, height],
                                 [1, 0, height],
                                 [0, 1, height],
                                 [1, 1, height]], dtype=np.int32)
        vertices = []
        for rel_xyz in relative_xyz:
            cube_vertex = grid_xyz + rel_xyz
            vertices.append(cube_vertex)
        # (N,8,3)
        vertices = np.stack(vertices, axis=1)
        # (N*8, 3)
        vertices = np.reshape(vertices, (-1, 3))
        print(type(vertices))
        return vertices

    def get_triangles(self, num_tri):
        cube_tris = np.array([[0, 4, 2], #side
                              [4, 6, 2],
                              [2, 6, 3],
                              [6, 7, 3],
                              [3, 7, 1],
                              [7, 5, 1],
                              [1, 5, 0],
                              [5, 4, 0],
                              #opposite direction
                              [2, 4, 0],
                              [2, 6, 4],
                              [3, 6, 2],
                              [3, 7, 6],
                              [1, 7, 3],
                              [1, 5, 7],
                              [0, 5, 1],
                              [0, 4, 5],
                              #top
                              [4, 5, 6],
                              [6, 5, 7],
                              #bottom
                              [2, 1, 0],        
                              [3, 1, 2],
                              ], dtype=np.int32)
        # (N, 8, 3)
        cube_tris = np.tile(cube_tris, (num_tri, 1, 1))
        offset = np.arange(0, num_tri * 8, 8).reshape((num_tri, 1, 1))
        cube_tris += offset
        return cube_tris

    def get_triangle_2d(self, num_tri):
        sqaure_tris = np.array([[0, 1, 2],
                               [1, 3, 2],
                               [2, 1, 0],
                               [2, 3, 1]], dtype=np.int32)
        sqaure_tris = np.tile(sqaure_tris, (num_tri, 1, 1))
        offset = np.arange(0, num_tri * 4, 4).reshape((num_tri, 1, 1))
        sqaure_tris += offset
        return sqaure_tris

    def draw_mesh_2d(self):
        # meshes = []
        # # for ctgr_idx, category in enumerate(self.categories):
        # #     vertices = self.vert_packs[ctgr_idx].copy()
        # #     vertices = np.reshape(vertices, (-1, 8, 3))
        # #     tri_2d_count = vertices.shape[0]
        # #     vertices = vertices[:, 0:4]
        # #     vertices = np.reshape(vertices, (-1, 3))
        # #     triangles = self.get_triangle_2d(tri_2d_count)
        # #     triangles = np.reshape(triangles, (-1, 3))
        # #     mesh = self.create_mesh_object(vertices, triangles, self.ctgr_colors[ctgr_idx])
        # #     meshes.append(mesh)
        # for ctgr_idx, category in enumerate(self.categories):
        #     if ctgr_idx == 0:
        #         continue
        #     vertices = self.vert_packs[ctgr_idx].copy()
        #     vertices = np.reshape(vertices, (-1, 8, 3))[:, 0:4]
        #     tri_2d_count = vertices.shape[0]
        #     vertices = np.reshape(vertices, (-1, 3))
        #     triangles = self.get_triangle_2d(tri_2d_count)
        #     triangles = np.reshape(triangles, (-1, 3))
        #     mesh = self.create_mesh_object(vertices, triangles, self.ctgr_colors[ctgr_idx])
        #     meshes.append(mesh)
        # # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[-0.5, -0.5, 0])
        # # meshes.append(mesh_frame)
        # o3d.visualization.draw_geometries(meshes)
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

    def draw_mesh_3d(self):
        meshes = []
        for ctgr_idx, category in enumerate(self.categories):
            if ctgr_idx == 0:
                continue
            vertices = self.vert_packs[ctgr_idx]
            triangles = self.tria_packs[ctgr_idx]
            triangles = np.reshape(triangles, (-1, 3))
            mesh = self.create_mesh_object(vertices, triangles, self.ctgr_colors[ctgr_idx])
            meshes.append(mesh)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[-0.5, -0.5, 0])
        meshes.append(mesh_frame)
        o3d.visualization.draw_geometries(meshes)

    def draw_ceiling(self):
        meshes = []
        for ctgr_idx, category in enumerate(self.categories):
            if ctgr_idx == 0:
                continue
            vertices = self.vert_packs[ctgr_idx].copy()
            vertices = np.reshape(vertices, (-1, 8, 3))[:, 0:4]
            tri_2d_count = vertices.shape[0]
            for vert in vertices:
                vert[:, 2] = 3
            vertices = np.reshape(vertices, (-1, 3))
            triangles = self.get_triangle_2d(tri_2d_count)
            triangles = np.reshape(triangles, (-1, 3))
            mesh = self.create_mesh_object(vertices, triangles, self.ctgr_colors[ctgr_idx])
            meshes.append(mesh)

    def create_mesh_object(self, vertices, triangles, color):
        vert = o3d.utility.Vector3dVector(vertices)
        tri = o3d.utility.Vector3iVector(triangles)
        mesh = o3d.geometry.TriangleMesh(vert, tri)
        print("Before", mesh)
        mesh = o3d.geometry.TriangleMesh.remove_duplicated_vertices(mesh)
        mesh = o3d.geometry.TriangleMesh.remove_duplicated_triangles(mesh)
        print("After", mesh)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh


if __name__ == "__main__":
    file = '/home/cheetah/lee_ws/3D-Map-Renderer/sample_map/sample_18_1.txt'
    Map = MapRenderer(file)


