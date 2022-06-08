import numpy as np


def update_mesh():
    # new_verts: (N, 3), new_trias: (N, 3)
    # mesh_verts: (M, 3), mesh_trias: (M, 3)
    new_verts = np.array([[1, 0], [3, 4], [5, 6], [5, 7]])
    new_trias = np.array([[0, 1, 2], [1, 2, 3], [3, 2, 1]])
    mesh_verts = np.array([[0, 0], [1, 2], [5, 6], [3, 4], [7, 8]])
    mesh_trias = np.array([[0, 1, 2], [3, 4, 5], [2, 4, 6]])

    mesh_verts_3d = mesh_verts.reshape((-1, 1, 2))     # (M, 1, 3)
    new_verts_3d = new_verts.reshape((1, -1, 2))       # (1, N, 3)
    match = (mesh_verts_3d == new_verts_3d).all(axis=-1)    # (M, N)
    print("mesh vs new match:\n", match)
    mesh_inds, new_inds = match.nonzero()
    print("match indices:", mesh_inds, new_inds)
    indices_map = np.ones((new_verts.shape[0]), dtype=int) * -1
    indices_map[new_inds] = mesh_inds
    print("existing vertex indices:", indices_map)
    invalid_indices = np.array((indices_map == -1).nonzero())[0]
    print("invalid_indices:", invalid_indices)
    indices_map[invalid_indices] = np.arange(start=mesh_verts.shape[0], stop=mesh_verts.shape[0]+invalid_indices.size)
    print("zero filled indices_map", indices_map)
    verts_to_add = new_verts[invalid_indices]
    print(mesh_verts.shape, verts_to_add.shape)
    mesh_verts = np.concatenate([mesh_verts, verts_to_add], axis=0)
    print("updated mesh vertices\n", mesh_verts)
    print("triangles before\n", new_trias)
    new_trias = indices_map[new_trias]
    print("updated triangles\n", new_trias)
    mesh_trias = np.concatenate([mesh_trias, new_trias], axis=0)
    print("total triangles\n", mesh_trias)


if __name__ == "__main__":
    update_mesh()
