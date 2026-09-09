import torch
from utils.log import logger


def explode_mesh_gpu(mesh, explode_coeff=1.0):
    """
    Applies an explosion effect to a mesh by offsetting its vertices along the surface normals.

    Args:
        mesh: The input mesh object. It can be either a trimesh.Trimesh or a chroma.geometry.Mesh.
        explode_coeff: The coefficient controlling the magnitude of the explosion effect. Default is 1.0.

    Returns:
        A dictionary containing the exploded mesh data. The dictionary has the following keys:
        - 'vertices': A numpy array of shape (N, 3) representing the exploded vertices.
        - 'faces' or 'triangles': A numpy array of shape (M, 3) representing the exploded faces or triangles,
          depending on the type of the input mesh object.
    """
    # Move data to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(mesh, "faces"):  # is trimesh.Trimesh
        faces = mesh.faces
    else:  # is chroma.geometry.Mesh
        faces = mesh.triangles

    # Convert mesh data to PyTorch tensors
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.int64, device=device)

    # Compute triangles
    tri = vertices[faces]

    # Compute normals like Chroma does
    v1, v2, v3 = tri[:, 0], tri[:, 1], tri[:, 2]
    normals = torch.linalg.cross(v2 - v1, v3 - v1, dim=1)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    # Offset triangles
    tri = tri + explode_coeff * normals.unsqueeze(1)

    # Flatten and reshape
    vertices_with_dupes = tri.reshape(-1, 3)
    faces_with_dupes = torch.arange(
        vertices_with_dupes.shape[0], device=device
    ).reshape(-1, 3)

    # Remove duplicates
    unique_vertices, inverse_indices = torch.unique(
        vertices_with_dupes, dim=0, return_inverse=True
    )
    faces = inverse_indices[faces_with_dupes.reshape(-1)].reshape(-1, 3)

    return dict(
        vertices=unique_vertices.cpu().numpy(),
        **(
            dict(faces=faces.cpu().numpy())
            if hasattr(mesh, "faces")
            else dict(triangles=faces.cpu().numpy())
        ),
    )


def main():
    import argparse

    import geometry.materials as mats
    import geometry.surfaces as surf
    from chroma.geometry import Geometry, Mesh, Solid
    from chroma.loader import create_geometry_from_obj, mesh_from_stl
    from utils.color import format_color
    from chroma.camera import Camera

    parser = argparse.ArgumentParser(
        description="Visualize inner and outer materials using chroma visualization"
    )
    parser.add_argument("input", type=str, help="Path to input mesh STL")
    args = parser.parse_args()

    logger.info(f"Loading mesh from {args.input}")
    mesh = mesh_from_stl(args.input)

    logger.info("Exploding mesh")
    logger.info("Note:\n\tmaterial1 is YELLOW, material2 is GREEN")
    inside = Solid(
        mesh,
        mats.steel,
        mats.steel,
        surf.steel,
        color=format_color("yellow", alpha=0.9),
    )

    outside = Solid(
        Mesh(**explode_mesh_gpu(mesh)),
        mats.steel,
        mats.steel,
        surf.steel,
        color=format_color("green", alpha=0.9),
    )

    geo = Geometry()
    geo.add_solid(outside)
    geo.add_solid(inside)
    geo = create_geometry_from_obj(geo)

    cam = Camera(geo)
    cam.run()


if __name__ == "__main__":
    main()
