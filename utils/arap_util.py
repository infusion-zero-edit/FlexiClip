import pydiffvg


def warp_svg(shapes, faces, face_index, bary_coord, vertices, cum_sizes):
    """
    vertices: new positions of mesh vertices
    """
    index = faces[face_index]
    new_pts = (vertices[index] * bary_coord[:, :, None]).sum(dim=1).float()

    new_shapes = [pydiffvg.Path(
        num_control_points=shape.num_control_points,
        points=new_pts[cum_sizes[i]:cum_sizes[i+1]],
        is_closed=shape.is_closed,
    ) for i, shape in enumerate(shapes)]
    return new_shapes
