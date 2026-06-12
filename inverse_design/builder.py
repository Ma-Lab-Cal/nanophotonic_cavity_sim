"""Differentiable Tidy3D simulation construction.

Mirrors ``main_code.simulation.Cavity_simulation`` geometry conventions
(waveguide slab + air-hole GeometryGroup, PML, y/z symmetry for Ey) but with:

* hole vertices traced by autograd (one traced Structure holding a
  GeometryGroup of PolySlabs),
* frequency-domain monitors only (ModeMonitor at the output port, point
  FieldMonitors at the atom positions, a closed box of 2D FieldMonitors for
  total radiated power, plus one non-objective field profile for plotting),
* a deterministic dipole source (no random phase).

The existing class is untouched; the non-gradient verification path lives in
``diagnostics.FrozenGeometryCavitySimulation``.
"""

from __future__ import annotations

import numpy as np
import tidy3d as td

from inverse_design import config as cfg
from inverse_design.parametrization import CavityParametrization

C0 = td.constants.C_0

# module-level cache: dispersion fitting is slow and medium is theta-independent
_MEDIUM_CACHE = {}


def resolve_medium(medium_spec):
    key = str(medium_spec)
    if key not in _MEDIUM_CACHE:
        from main_code.simulation import Cavity_simulation
        if isinstance(medium_spec, (int, float)):
            _MEDIUM_CACHE[key] = td.Medium(permittivity=float(medium_spec) ** 2)
        else:
            _MEDIUM_CACHE[key] = Cavity_simulation._medium_from_file(medium_spec)
    return _MEDIUM_CACHE[key]


def n_eff_guess(context) -> float:
    """Refractive index of the beam at f0 (upper bound for target_neff)."""
    medium = resolve_medium(context["medium"])
    eps = medium.eps_model(context["freq0"])
    return float(np.real(np.sqrt(eps)))


def sim_domain(positions_np, a_np, context, padding_factor=2.0):
    wlM = C0 / (context["freq0"] - 2e12)
    padding = padding_factor * wlM
    xmin = positions_np[0] - a_np[0] - padding
    xmax = positions_np[-1] + a_np[-1] + padding
    center = ((xmin + xmax) / 2.0, 0.0, 0.0)
    size = (xmax - xmin,
            context["width"] + 2 * padding,
            context["thickness"] + 2 * padding)
    return center, size, wlM


def build_structures(layout, par: CavityParametrization, sim_size_x: float):
    """[untraced slab, traced hole group] as td.Structures."""
    context = par.context
    thickness = context["thickness"]
    width = context["width"]
    medium = resolve_medium(context["medium"])

    slab = td.Structure(
        geometry=td.PolySlab(
            vertices=[[-2 * sim_size_x, width / 2], [2 * sim_size_x, width / 2],
                      [2 * sim_size_x, -width / 2], [-2 * sim_size_x, -width / 2]],
            axis=2, slab_bounds=(-thickness / 2, thickness / 2),
        ),
        medium=medium, name="Nanobeam",
    )

    import autograd.numpy as anp
    hole_geoms = []
    for i in range(par.n_holes):
        v = layout["vertices"][i]
        verts = anp.stack([v[:, 0] + layout["positions"][i], v[:, 1]], axis=1)
        hole_geoms.append(td.PolySlab(
            vertices=verts, axis=2,
            slab_bounds=(-thickness / 2, thickness / 2)))

    holes = td.Structure(
        geometry=td.GeometryGroup(geometries=hole_geoms),
        medium=td.Medium(permittivity=1.0), name="Nanobeam_crystals",
    )
    return [slab, holes]


def source_position(par: CavityParametrization, positions_np) -> float:
    """Dipole x at the central dielectric antinode (detached scalar)."""
    ic = par._center_left_idx
    if par.n_cells["N_defect"] % 2 == 0:
        return float((positions_np[ic] + positions_np[ic + 1]) / 2.0)
    # odd: antinode midway to the next hole (legacy convention: +a/2)
    return float((positions_np[ic] + positions_np[ic + 1]) / 2.0)


def build_forward_simulation(theta, par: CavityParametrization,
                             sim_cfg: cfg.SimConfig) -> td.Simulation:
    context = par.context
    layout = par.layout(theta)
    lay_np = CavityParametrization.detach(layout)
    positions_np = np.asarray(lay_np["positions"])
    a_np = np.asarray(lay_np["a"])

    center, size, wlM = sim_domain(positions_np, a_np, context)
    structures = build_structures(layout, par, size[0])

    freqs = list(sim_cfg.freqs)
    x_src = source_position(par, positions_np)
    thickness = context["thickness"]

    source = td.PointDipole(
        center=(x_src, 0, 0), name="Point_Dipole_Source",
        polarization=context["polarization"],
        source_time=td.GaussianPulse(freq0=sim_cfg.f_center,
                                     fwidth=sim_cfg.fwidth_source, phase=0.0),
    )

    monitors = []

    # output waveguide mode monitor (left of all holes)
    x_wg = positions_np[0] - a_np[0] - 1.0
    monitors.append(td.ModeMonitor(
        center=(x_wg, 0, 0),
        size=(0, size[1] - wlM, size[2] - wlM),
        freqs=freqs, name="wg_out",
        mode_spec=td.ModeSpec(num_modes=1, target_neff=n_eff_guess(context)),
    ))

    # atom-position point monitors above the top surface
    for k, dz in enumerate(cfg.ATOM_SURFACE_OFFSETS_UM):
        monitors.append(td.FieldMonitor(
            center=(x_src, 0, thickness / 2 + dz), size=(0, 0, 0),
            freqs=freqs, fields=["Ex", "Ey", "Ez"], name=f"atom_{k}",
        ))

    # closed box of 2D field monitors for total radiated power
    for axis_idx, axis in enumerate("xyz"):
        for sign in ("-", "+"):
            mcenter = list(center)
            mcenter[axis_idx] += (1 if sign == "+" else -1) * (size[axis_idx] / 2 - wlM / 2)
            msize = list(size)
            msize[axis_idx] = 0.0
            monitors.append(td.FieldMonitor(
                center=tuple(mcenter), size=tuple(msize), freqs=freqs,
                fields=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
                name=f"box_{sign}{axis}",
            ))

    # diagnostic-only field profile (single frequency, not used in the loss)
    monitors.append(td.FieldMonitor(
        center=(center[0], 0, 0), size=(td.inf, td.inf, 0),
        freqs=[sim_cfg.f_center], fields=["Ex", "Ey", "Ez"], name="profile_z",
    ))

    mesh_override = td.MeshOverrideStructure(
        geometry=td.Box(center=(0, 0, 0),
                        size=(0.6 * (positions_np[-1] - positions_np[0]),
                              context["width"], thickness)),
        dl=(sim_cfg.defect_mesh_dl,) * 3,
    )

    # Ey polarization -> (1,-1,1); x-symmetry off (one-sided cavity)
    sym_map = {"Ex": [-1, 1, 1], "Ey": [1, -1, 1], "Ez": [1, 1, -1]}
    sym = sym_map[context["polarization"]]
    sym[0] = 0

    return td.Simulation(
        center=center, size=size, structures=structures, sources=[source],
        monitors=monitors, run_time=sim_cfg.run_time,
        boundary_spec=td.BoundarySpec(x=td.Boundary.pml(), y=td.Boundary.pml(),
                                      z=td.Boundary.pml()),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=sim_cfg.min_steps_per_wvl,
                                   override_structures=[mesh_override]),
        symmetry=tuple(sym),
        medium=td.Medium(permittivity=1.0),
        shutoff=sim_cfg.shutoff,
    )
