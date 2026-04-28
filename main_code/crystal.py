import tidy3d as td
import numpy as np

def crystal_polygon_2d(context, params, lattice, n_pts=128):
    """Return an (N, 2) array of xy-vertices for a crystal centered at the origin.
 
    This is the single source of truth for crystal shapes. Both the 3-D
    Tidy3D geometry (``crystal_geometry``) and any 2-D rendering / GDS
    export should derive from the same shape definitions here.
 
    Parameters
    ----------
    geometry : str
        Crystal type (must match a case below).
    params : array-like
        Shape parameters whose meaning depends on *geometry*.
    n_pts : int
        Vertex count for curved shapes.
 
    Returns
    -------
    np.ndarray, shape (N, 2)
    """
    hp = np.atleast_1d(params)

    geometry = context["geometry"]

    if geometry == "rectangle":
        hx, hy = hp[0] / 2, hp[1] / 2
        return np.array([
            [-hx, -hy],
            [ hx, -hy],
            [ hx,  hy],
            [-hx,  hy],
            [-hx, -hy],
        ])

    if geometry == "ellipse":
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
        rx, ry = hp[0] / 2, hp[1] / 2
        return np.column_stack([rx * np.cos(theta), ry * np.sin(theta)])
 
    if geometry == "sawtooth_square":
        # params: [tooth_w, tooth_h]
        # Simple rectangle for one tooth
        tw, th = hp[0], hp[1]
        nw = context["width"]

        return np.array([[-tw/2, nw/2+th], [tw/2, nw/2+th], [tw/2, -nw/2-th], [-tw/2, -nw/2-th]])

    if geometry == "sawtooth_round":
        # params: [tooth_w, tooth_h, fillet_diameter]
        tw, th = hp[0], hp[1]
        r = hp[2] / 2
        nw = context["width"]

        quarter = n_pts // 4
    
        def arc(cx, cy, r, a0, a1, n):
            angles = np.linspace(a0, a1, n)
            return np.column_stack([cx + r * np.cos(angles),
                                    cy + r * np.sin(angles)])
    
        pts = []

        check_h = th - 2 * r
        check_w = tw - 2 * r

        if check_h >= 0 and check_w >= 0:

            pts.append([-lattice/2, nw/2])

            # Bottom-left: concave fillet — 3π/2 → 2π
            pts.append(arc(-tw/2 - r, nw/2 + r, r, 3*np.pi/2, 2*np.pi, quarter))
            # Top-left: convex round — π → π/2
            pts.append(arc(-tw/2 + r, nw/2 + th - r, r, np.pi, np.pi/2, quarter))
            # Top-right: convex round — π/2 → 0
            pts.append(arc(tw/2 - r, nw/2 + th - r, r, np.pi/2, 0, quarter))
            # Bottom-right: concave fillet — π → 3π/2
            pts.append(arc(tw/2 + r, nw/2 + r, r, np.pi, 3*np.pi/2, quarter))
            pts.append([lattice/2, nw/2])

        else: 
            r = min(th / 2, tw / 2)

            pts.append([-lattice/2, nw/2])

            # Bottom-left: concave fillet — 3π/2 → 2π
            pts.append(arc(-tw/2 - r, nw/2 + r, r, 3*np.pi/2, 2*np.pi, quarter))
            # Top-left: convex round — π → π/2
            pts.append(arc(-tw/2 + r, nw/2 + th - r, r, np.pi, np.pi/2, quarter))
            # Top-right: convex round — π/2 → 0
            pts.append(arc(tw/2 - r, nw/2 + th - r, r, np.pi/2, 0, quarter))
            # Bottom-right: concave fillet — π → 3π/2
            pts.append(arc(tw/2 + r, nw/2 + r, r, np.pi, 3*np.pi/2, quarter))
            pts.append([lattice/2, nw/2])

        verts_top = np.vstack(pts)
        verts_bot = verts_top.copy()
        verts_bot[:, 1] *= -1
        verts_bot = verts_bot[::-1]
        verts = np.vstack([verts_top, verts_bot])
        return verts


    raise ValueError(f"Unknown geometry: '{geometry}'")
 
 
def crystal_tidy(context, center, params, lattice):
    """Create a single 3-D Tidy3D crystal geometry at the given center.
 
    Uses ``crystal_polygon_2d`` for all shapes to keep the geometry
    definition in one place.
    """

    thickness = context["thickness"]

    verts = crystal_polygon_2d(context, params, lattice)
    verts_shifted = verts + np.array([center[0], center[1]])
 
    return td.PolySlab(
        vertices=verts_shifted,
        slab_bounds=(
            -thickness / 2 + center[2],
             thickness / 2 + center[2],
        ),
        axis=2,
    )


