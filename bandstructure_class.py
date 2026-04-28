import meep as mp
import numpy as np
from meep import mpb
import matplotlib.pyplot as plt
import json
import os

C0 = 2.99792458e8  # m/s
from itertools import product as iterproduct

class BandStructureSim:
    """
    MPB band structure solver. All inputs in µm.
    
    Parameters
    ----------
    Lx : float
        Lattice constant in x (µm). Used as normalisation unit.
    Ly : float
        Lattice constant in y (µm).
    thickness : float
        Slab thickness (µm).
    geometry : str
        'circular', 'square', or 'ellipse'.
    geometry_params : list of float
        In µm. diameter for circular, [dx, dy] for ellipse/square.
    material_index : float
        Refractive index of the slab.
    num_bands : int
    resolution : int
    """

    def __init__(self, Lx, Ly, thickness,
                 geometry, geometry_params, material_index, factor_y=5, factor_z=10,
                 num_bands=5, resolution=32):

        self.Lx = Lx
        self.Ly = Ly
        self.thickness = thickness
        self.geometry_params = geometry_params
        self.material_index = material_index
        self.geometry_type = geometry
        self.num_bands = num_bands
        self.resolution = resolution

        # Normalised values (units of Lx)
        self.Ly_norm = Ly / Lx
        self.t_norm = thickness / Lx
        self.geometry_params_norm = [p / Lx for p in geometry_params]

        self.factor_y = factor_y
        self.factor_z = factor_z

        self.geometry = self._build_geometry(self.geometry_params_norm)

        # Results
        self.kpts = None
        self.kpar = None
        self.zeven_freqs = None
        self.zodd_freqs = None
        self.zeven_gaps = None
        self.zodd_gaps = None
        self.ms = None
        self.sweep_data = None

    def init_solver(self):
        """Instantiate the ModeSolver (needed for plot_epsilon) without running bands."""
        self.ms = mpb.ModeSolver(
            geometry=self.geometry,
            geometry_lattice=self.lattice,
            k_points=[mp.Vector3(0, 0, 0)],
            resolution=self.resolution,
            num_bands=self.num_bands,
            default_material=mp.Medium(epsilon=1),
            verbose=False,
        )
        self.ms.init_params(mp.NO_PARITY, False)

    def _build_geometry(self, geometry_params_norm):
        mat = mp.Medium(index=self.material_index)
        slab = mp.Block(
            material=mat,
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(mp.inf, self.Ly_norm, self.t_norm),
        )

        if self.geometry_type == "circular":
            hole = mp.Cylinder(
                material=mp.air,
                center=mp.Vector3(0, 0, 0),
                radius=geometry_params_norm[0]/2,
                height=mp.inf,
                axis=mp.Vector3(0, 0, 1),
            )
            self.lattice = mp.Lattice(size=mp.Vector3(1, self.factor_y*self.Ly_norm, self.factor_z*self.t_norm),basis1=mp.Vector3(1, 0, 0))
            return [slab, hole]
        
        elif self.geometry_type == "square":
            hole = mp.Block(
                material=mp.air,
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(geometry_params_norm[0], geometry_params_norm[1], mp.inf),
            )
            self.lattice = mp.Lattice(size=mp.Vector3(1, self.factor_y*self.Ly_norm, self.factor_z*self.t_norm),basis1=mp.Vector3(1, 0, 0))
            return [slab, hole]
        
        elif self.geometry_type == "ellipse":
            hole = mp.Ellipsoid(
                material=mp.air,
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(geometry_params_norm[0], geometry_params_norm[1], mp.inf),
            )    
            self.lattice = mp.Lattice(size=mp.Vector3(1, self.factor_y*self.Ly_norm, self.factor_z*self.t_norm),basis1=mp.Vector3(1, 0, 0))
            return [slab, hole]
        
        elif self.geometry_type == "sawtooth_square":
            # Central beam (same as slab)

            tooth_w = geometry_params_norm[0]  # width along x
            tooth_h = geometry_params_norm[1]  # height protruding along y

            # Top tooth: centered at y = +Ly_norm/2 + tooth_h/2
            top_tooth = mp.Block(
                material=mat,
                center=mp.Vector3(0, self.Ly_norm / 2 + tooth_h / 2, 0),
                size=mp.Vector3(tooth_w, tooth_h, self.t_norm),
            )
            # Bottom tooth: centered at y = -Ly_norm/2 - tooth_h/2
            bot_tooth = mp.Block(
                material=mat,
                center=mp.Vector3(0, -self.Ly_norm / 2 - tooth_h / 2, 0),
                size=mp.Vector3(tooth_w, tooth_h, self.t_norm),
            )

            self.lattice = mp.Lattice(size=mp.Vector3(1, self.factor_y*(self.Ly_norm+2*tooth_h), self.factor_z*self.t_norm), basis1=mp.Vector3(1, 0, 0))
            return [slab, top_tooth, bot_tooth]

        elif self.geometry_type == "sawtooth_round":
            tooth_w = geometry_params_norm[0]
            tooth_h = geometry_params_norm[1]
            r       = geometry_params_norm[2] / 2

            check_h = tooth_h - 2 * r
            check_w = tooth_w - 2 * r
            if not (check_h >= 0 and check_w >= 0):
                r = min(tooth_h / 2, tooth_w / 2)
                print(f"Adjusting radius to {r:.3f}.")

            n_pts   = 32        # points per quarter-arc
            lattice = self.Lx  # or whatever your lattice constant variable is

            def arc(cx, cy, r, a0, a1, n):
                angles = np.linspace(a0, a1, n)
                return np.column_stack([cx + r * np.cos(angles),
                                        cy + r * np.sin(angles)])

            nw = self.Ly_norm
            pts = []

            pts.append([-lattice / 2,  nw / 2])
            pts.append(arc(-tooth_w/2 - r,  nw/2 + r,          r, 3*np.pi/2, 2*np.pi, n_pts))  # concave fillet BL
            pts.append(arc(-tooth_w/2 + r,  nw/2 + tooth_h - r, r, np.pi,     np.pi/2, n_pts))  # convex round TL
            pts.append(arc( tooth_w/2 - r,  nw/2 + tooth_h - r, r, np.pi/2,   0,       n_pts))  # convex round TR
            pts.append(arc( tooth_w/2 + r,  nw/2 + r,           r, np.pi,     3*np.pi/2, n_pts)) # concave fillet BR
            pts.append([ lattice / 2,  nw / 2])

            verts_top = np.vstack(pts)
            verts_bot = verts_top.copy()
            verts_bot[:, 1] *= -1
            verts_bot = verts_bot[::-1]

            all_verts = np.vstack([verts_top, verts_bot])

            # mp.Prism takes a list of Vector3 in the xy-plane, extruded along z
            prism_verts = [mp.Vector3(x, y, 0) for x, y in all_verts]

            geom = [
                slab,
                mp.Prism(
                    vertices=prism_verts,
                    height=self.t_norm,
                    axis=mp.Vector3(0, 0, 1),
                    center=mp.Vector3(0, 0, 0),
                    material=mat,
                )
            ]

            self.lattice = mp.Lattice(size=mp.Vector3(1, self.factor_y*(self.Ly_norm+2*tooth_h), self.factor_z*self.t_norm), basis1=mp.Vector3(1, 0, 0))
            return geom
        else:
            raise ValueError(f"Unknown geometry '{self.geometry_type}'.")


    # ──────────────────────────────────────────────────────────────────
    # Unit conversions
    # ──────────────────────────────────────────────────────────────────

    def freq_norm_to_Hz(self, freq_norm):
        return freq_norm * C0 / (self.Lx * 1e-6)

    def freq_norm_to_THz(self, freq_norm):
        return freq_norm * C0 / (self.Lx * 1e-6) * 1e-12

    def freq_Hz_to_norm(self, freq_Hz):
        return freq_Hz * (self.Lx * 1e-6) / C0

    # ──────────────────────────────────────────────────────────────────
    # Run simulations
    # ──────────────────────────────────────────────────────────────────

    def _run_solver(self, k_points):
        ms = mpb.ModeSolver(
            geometry=self.geometry,
            geometry_lattice=self.lattice,
            k_points=k_points,
            resolution=self.resolution,
            num_bands=self.num_bands,
            default_material=mp.Medium(epsilon=1), 
            verbose=True
        )

        ms.run_zeven()
        self.zeven_freqs = np.array(ms.all_freqs)
        self.zeven_gaps = ms.gap_list

        ms.run_zodd()
        self.zodd_freqs = np.array(ms.all_freqs)
        self.zodd_gaps = ms.gap_list

        self.kpts = np.array([[k.x, k.y, k.z] for k in ms.k_points])
        self.kpar = np.linalg.norm(self.kpts[:, :2], axis=1)
        self.ms = ms

        return self.kpts, self.kpar, self.zeven_freqs, self.zodd_freqs

    def run_at_k(self, kx=0.5):
        return self._run_solver([mp.Vector3(kx, 0, 0)])

    # ──────────────────────────────────────────────────────────────────
    # Run and Plot bandstructure
    # ──────────────────────────────────────────────────────────────────

    def run_bandstructure(self, k_min=0.0, k_max=0.5, n_k=8):
        k_points = mp.interpolate(n_k, [
            mp.Vector3(k_min, 0, 0),
            mp.Vector3(k_max, 0, 0),
        ])
        return self._run_solver(k_points)

    def plot_bands(self, units='norm', freq0_Hz=None, ylim=None, title=None):
        """
        Plot band structure.
        
        Parameters
        ----------
        units : str
            'norm' → frequency in c/a
            'THz'  → frequency in THz
        freq0_Hz : float, optional
            Draw a horizontal line at this frequency.
        ylim : tuple, optional
        """
        if self.zeven_freqs is None:
            raise RuntimeError("No data. Run run_bandstructure() first.")

        fig, ax = plt.subplots(figsize=(10, 6))

        if units == 'norm':
            zeven_y = self.zeven_freqs
            zodd_y = self.zodd_freqs
            light_y = self.kpar
            ylabel = 'Frequency (c/a)'
            default_ylim = (0, 0.5)
            freq0_line = self.freq_Hz_to_norm(freq0_Hz) if freq0_Hz else None
        elif units == 'THz':
            zeven_y = self.freq_norm_to_THz(self.zeven_freqs)
            zodd_y = self.freq_norm_to_THz(self.zodd_freqs)
            light_y = self.freq_norm_to_THz(self.kpar)
            ylabel = 'Frequency (THz)'
            default_ylim = (0, self.freq_norm_to_THz(0.5))
            freq0_line = freq0_Hz * 1e-12 if freq0_Hz else None
        else:
            raise ValueError(f"Unknown units '{units}'. Use 'norm' or 'THz'.")

        ax.plot(self.kpar, zeven_y, color='red', linewidth=1.2)
        ax.plot(self.kpar, zodd_y, color='blue', linewidth=1.2)
        ax.plot([], [], color='red', linewidth=1.2, label='TE-like (zeven)')
        ax.plot([], [], color='blue', linewidth=1.2, label='TM-like (zodd)')
        ax.plot(self.kpar, light_y, 'k--', alpha=0.4, label='Light line')

        if freq0_line is not None:
            ax.axhline(freq0_line, color='black', linewidth=0.8, label='Target freq')

        ax.set_xlabel('k∥ (2π/a)', size=14)
        ax.set_ylabel(ylabel, size=14)
        ax.set_ylim(ylim or default_ylim)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(title or 'Band Structure')
        plt.tight_layout()
        plt.show()

        self._print_gaps()
        return fig, ax

    def _print_gaps(self):
        for label, gaps in [('TE-like (zeven)', self.zeven_gaps),
                            ('TM-like (zodd)', self.zodd_gaps)]:
            if not gaps:
                continue
            print(f"{label} gaps:")
            for gap in gaps:
                if gap[0] > 0:
                    f_lo_THz = self.freq_norm_to_THz(gap[1])
                    f_hi_THz = self.freq_norm_to_THz(gap[2])
                    print(f"  Gap {gap[0]:.1f}%: {gap[1]:.4f} – {gap[2]:.4f} (c/a) "
                          f"= {f_lo_THz:.1f} – {f_hi_THz:.1f} THz")

    # ──────────────────────────────────────────────────────────────────
    # Run and Plot sweep
    # ──────────────────────────────────────────────────────────────────


    def run_sweep(self, param_name, param_values_um, kx=0.5):
        all_zeven = []
        all_zodd = []
        all_Lx = []

        original_geometry_params = list(self.geometry_params)
        original_Ly = self.Ly
        original_thickness = self.thickness
        original_Lx = self.Lx

        for val in param_values_um:
            if param_name == 'p1':
                self.geometry_params[0] = val
            elif param_name == 'p2':
                self.geometry_params[1] = val
            elif param_name == 'Ly':
                self.Ly = val
            elif param_name == 'thickness':
                self.thickness = val
            elif param_name == 'Lx':
                self.Lx = val
            else:
                raise ValueError(f"Unknown param_name '{param_name}'.")
            

            self.Ly_norm = self.Ly / self.Lx
            self.t_norm = self.thickness / self.Lx
            self.geometry_params_norm = [p / self.Lx for p in self.geometry_params]

            self.lattice = mp.Lattice(
                size=mp.Vector3(1, self.factor_y*self.Ly_norm, self.factor_z*self.t_norm),
                basis1=mp.Vector3(1, 0, 0)
            )

            self.geometry = self._build_geometry(self.geometry_params_norm)
            self._run_solver([mp.Vector3(kx, 0, 0)])
            all_Lx.append(self.Lx)
            all_zeven.append(self.zeven_freqs.copy())
            all_zodd.append(self.zodd_freqs.copy())

        # Restore originals
        self.Lx = original_Lx
        self.geometry_params = original_geometry_params
        self.geometry_params_norm = [p / original_Lx for p in original_geometry_params]
        self.Ly = original_Ly
        self.Ly_norm = original_Ly / self.Lx
        self.thickness = original_thickness
        self.t_norm = original_thickness / original_Lx
        self.geometry = self._build_geometry(self.geometry_params_norm)

        self.sweep_data = {
            'param_name': param_name,
            'param_values_um': np.array(param_values_um),
            'kx': kx,
            'zeven_freqs': np.squeeze(np.array(all_zeven)),
            'zodd_freqs': np.squeeze(np.array(all_zodd)),
            'Lx_per_point': np.array(all_Lx)
        }
        return self.sweep_data

    def plot_sweep(self, units='norm', freq0_Hz=None, ylim=None, title=None):
        """
        Plot bands vs swept parameter.
        
        Parameters
        ----------
        units : str
            'norm' or 'THz'.
        freq0_Hz : float, optional
            Draw a horizontal line at this frequency.
        ylim : tuple, optional
        """
        if self.sweep_data is None:
            raise RuntimeError("No sweep data. Call run_sweep() first.")

        sd = self.sweep_data
        x = sd['param_values_um'] * 1e3  # nm for x-axis
        fig, ax = plt.subplots(figsize=(10, 6))

        for band_idx in range(sd['zeven_freqs'].shape[1]):
            freqs = sd['zeven_freqs'][:, band_idx]
            if units == 'THz':
                Lx_arr = sd['Lx_per_point']  # shape (n_points, 1) for broadcasting
                y = freqs * C0 / (Lx_arr * 1e-6) * 1e-12
            else:
                y = freqs
            ax.plot(x, y, 'r.-', markersize=4, linewidth=0.8,
                    label='TE-like' if band_idx == 0 else None)

        for band_idx in range(sd['zodd_freqs'].shape[1]):
            freqs = sd['zodd_freqs'][:, band_idx]
            if units == 'THz':
                Lx_arr = sd['Lx_per_point']  # shape (n_points, 1) for broadcasting
                y = freqs * C0 / (Lx_arr * 1e-6) * 1e-12
            else:
                y = freqs
            ax.plot(x, y, 'b.--', markersize=4, linewidth=0.8,
                    label='TM-like' if band_idx == 0 else None)

        if freq0_Hz is not None:
            if units == 'THz':
                ax.axhline(freq0_Hz * 1e-12, color='green', ls='--', alpha=0.6,
                           label='Target freq')
            else:
                ax.axhline(self.freq_Hz_to_norm(freq0_Hz), color='green', ls='--',
                           alpha=0.6, label='Target freq')

        ylabel = 'Frequency (THz)' if units == 'THz' else 'Frequency (c/a)'
        ax.set_xlabel(f"{sd['param_name']} (nm)", size=14)
        ax.set_ylabel(ylabel, size=14)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend()
        ax.grid(True, alpha=0.3)
        # With this:
        if title is None:
            # Map all parameters to display names and values (in nm)
            all_params = {
                'Lx':        ('Lx', self.Lx * 1e3),
                'Ly':        ('Ly', self.Ly * 1e3),
                'thickness': ('t',  self.thickness * 1e3),
                'p1':        (f'{self.geometry_type} p1', self.geometry_params[0] * 1e3),
            }
            if len(self.geometry_params) > 1:
                all_params['p2'] = (f'{self.geometry_type} p2', self.geometry_params[1] * 1e3)

            # Build string from non-swept params
            fixed = [f"{label}={val:.0f} nm"
                    for key, (label, val) in all_params.items()
                    if key != sd['param_name']]
            title = f"Bands vs {sd['param_name']} | kx={sd['kx']} | " + ", ".join(fixed)

        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return fig, ax


    # ──────────────────────────────────────────────────────────────────
    # Run and Plot multi sweep
    # ──────────────────────────────────────────────────────────────────

    def run_multisweep(self, param_ranges, kx=0.5):
        """
        Sweep over a grid of multiple parameters simultaneously.

        Parameters
        ----------
        param_ranges : dict
            Keys are parameter names ('Lx', 'Ly', 'rx', 'ry', 'thickness'),
            values are array-like of values in µm.
            Example: {'Lx': [0.3, 0.35, 0.4], 'rx': [0.05, 0.07, 0.09]}
        kx : float
            Fixed k-point for the sweep.

        Returns
        -------
        dict with keys:
            'param_names'  : list of str
            'param_grids'  : dict mapping name → 1-D array of swept values (µm)
            'grid_shape'   : tuple, shape of the N-D parameter grid
            'kx'           : float
            'zeven_freqs'  : np.ndarray, shape (*grid_shape, 1, num_bands)
            'zodd_freqs'   : np.ndarray, shape (*grid_shape, 1, num_bands)
        """
        param_names = list(param_ranges.keys())
        param_arrays = [np.atleast_1d(param_ranges[k]) for k in param_names]
        grid_shape = tuple(len(a) for a in param_arrays)

        # Save originals
        original = {
            'Lx': self.Lx,
            'Ly': self.Ly,
            'thickness': self.thickness,
            'geometry_params': list(self.geometry_params),
        }

        all_zeven = []
        all_zodd = []

        for combo in iterproduct(*param_arrays):
            # Apply each parameter
            for name, val in zip(param_names, combo):
                if name == 'p1':
                    self.geometry_params[0] = val
                elif name == 'p2':
                    self.geometry_params[1] = val
                elif name == 'Ly':
                    self.Ly = val
                elif name == 'thickness':
                    self.thickness = val
                elif name == 'Lx':
                    self.Lx = val
                else:
                    raise ValueError(f"Unknown param_name '{name}'.")

            # Recompute all normalised quantities
            self.Ly_norm = self.Ly / self.Lx
            self.t_norm = self.thickness / self.Lx
            self.geometry_params_norm = [p / self.Lx for p in self.geometry_params]
            self.lattice = mp.Lattice(
                size=mp.Vector3(1, self.factor_y*self.Ly_norm, self.factor_z*self.t_norm),
                basis1=mp.Vector3(1, 0, 0)
                )
            self.geometry = self._build_geometry(self.geometry_params_norm)

            self._run_solver([mp.Vector3(kx, 0, 0)])
            all_zeven.append(self.zeven_freqs.copy())
            all_zodd.append(self.zodd_freqs.copy())

        # Restore originals
        self.Lx = original['Lx']
        self.Ly = original['Ly']
        self.thickness = original['thickness']
        self.geometry_params = original['geometry_params']
        self.Ly_norm = self.Ly / self.Lx
        self.t_norm = self.thickness / self.Lx
        self.geometry_params_norm = [p / self.Lx for p in self.geometry_params]
        self.lattice = mp.Lattice(
            size=mp.Vector3(1, self.factor_y*self.Ly_norm, self.factor_z*self.t_norm),
            basis1=mp.Vector3(1, 0, 0)
            )
        self.geometry = self._build_geometry(self.geometry_params_norm)

        # Reshape: iterproduct iterates last index fastest → C-order reshape
        zeven = np.array(all_zeven).reshape(*grid_shape, 1, self.num_bands)
        zodd = np.array(all_zodd).reshape(*grid_shape, 1, self.num_bands)

        self.multisweep_data = {
            'param_names': param_names,
            'param_grids': {k: np.array(v) for k, v in zip(param_names, param_arrays)},
            'grid_shape': grid_shape,
            'kx': kx,
            'zeven_freqs': zeven,
            'zodd_freqs': zodd,
        }
        return self.multisweep_data
    
    def plot_multisweep(self, band_idx=0, polarization='zeven', units='norm',
                        freq0_Hz=None, title=None):
        md = self.multisweep_data
        names = md['param_names']

        if len(names) != 2:
            raise ValueError(
                f"plot_multisweep only supports 2-parameter sweeps, "
                f"got {len(names)}: {names}"
            )

        freqs_key = f'{polarization}_freqs'
        freqs = md[freqs_key][:, :, 0, band_idx]

        if units == 'THz':
            if 'Lx' in names:
                lx_idx = names.index('Lx')
                lx_vals = md['param_grids']['Lx']
                shape = [1, 1]
                shape[lx_idx] = len(lx_vals)
                lx_broadcast = lx_vals.reshape(shape)
                freqs_plot = freqs * C0 / (lx_broadcast * 1e-6) * 1e-12
            else:
                freqs_plot = self.freq_norm_to_THz(freqs)
            ylabel = f'Band {band_idx} freq (THz)'
        else:
            freqs_plot = freqs
            ylabel = f'Band {band_idx} freq (c/a)'

        if freq0_Hz is not None:
            if units == 'THz':
                freqs_plot = freqs_plot - freq0_Hz * 1e-12
                ylabel = f'f - f_target (THz)'
            else:
                if 'Lx' in names:
                    lx_idx = names.index('Lx')
                    lx_vals = md['param_grids']['Lx']
                    shape = [1, 1]
                    shape[lx_idx] = len(lx_vals)
                    lx_broadcast = lx_vals.reshape(shape)
                    target_norm = freq0_Hz * (lx_broadcast * 1e-6) / C0
                else:
                    target_norm = self.freq_Hz_to_norm(freq0_Hz)
                freqs_plot = freqs_plot - target_norm
                ylabel = f'f - f_target (c/a)'

        x = md['param_grids'][names[1]] * 1e3
        y = md['param_grids'][names[0]] * 1e3

        # Symmetric color limits so white sits at zero
        vmax = np.max(np.abs(freqs_plot))
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x, y, freqs_plot, shading='auto',
                        cmap='RdBu', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=ylabel)
        ax.set_xlabel(f'{names[1]} (nm)', size=13)
        ax.set_ylabel(f'{names[0]} (nm)', size=13)
        pol_label = 'TE-like' if polarization == 'zeven' else 'TM-like'
        ax.set_title(title or f'{pol_label} band {band_idx} — kx={md["kx"]}')
        plt.tight_layout()
        plt.show()
        return fig, ax

    # ──────────────────────────────────────────────────────────────────
    # Epsilon visualisation
    # ──────────────────────────────────────────────────────────────────

    def plot_epsilon(self, periods=3, show_unitcell=True):
        if self.ms is None:
            self.init_solver()
        md = mpb.MPBData(rectify=True, periods=periods, resolution=self.resolution)
        eps = self.ms.get_epsilon()
        converted_eps = md.convert(eps)

        nx, ny, nz = converted_eps.shape

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # XY plane (z = 0)
        axes[0].imshow(
            converted_eps[:, :, nz // 2].T,
            interpolation='spline36', cmap='binary',
            origin='lower',
        )
        axes[0].set_title('XY cross-section (z = 0)')

        if show_unitcell:
            # The image spans 'periods' unit cells in each direction.
            # One unit cell in pixel coords is nx/periods wide, ny/periods tall.
            cell_w = nx / periods
            cell_h = ny / periods
            # Center of the image
            cx, cy = nx / 2, ny / 2

            import matplotlib.patches as patches

            # XY cell boundary
            rect_xy = patches.Rectangle(
                (cx - cell_w / 2, cy - cell_h / 2), cell_w, cell_h,
                linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--',
            )
            axes[0].add_patch(rect_xy)

        axes[0].axis('off')

        # YZ plane (x = 0)
        axes[1].imshow(
            converted_eps[nx // 2, :, :].T,
            interpolation='spline36', cmap='binary',
            aspect='auto', origin='lower',
        )
        axes[1].set_title('YZ cross-section (x = 0)')

        if show_unitcell:
            cell_hy = ny / periods
            cell_hz = nz / periods
            cy_yz, cz_yz = ny / 2, nz / 2
            rect_yz = patches.Rectangle(
                (cy_yz - cell_hy / 2, cz_yz - cell_hz / 2), cell_hy, cell_hz,
                linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--',
            )
            axes[1].add_patch(rect_yz)

        axes[1].axis('off')

        # XZ plane (y = 0)
        axes[2].imshow(
            converted_eps[:, ny // 2, :].T,
            interpolation='spline36', cmap='binary',
            aspect='auto', origin='lower',
        )
        axes[2].set_title('XZ cross-section (y = 0)')

        if show_unitcell:
            cell_wx = nx / periods
            cell_wz = nz / periods
            cx_xz, cz_xz = nx / 2, nz / 2
            rect_xz = patches.Rectangle(
                (cx_xz - cell_wx / 2, cz_xz - cell_wz / 2), cell_wx, cell_wz,
                linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--',
            )
            axes[2].add_patch(rect_xz)

        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
        return fig, axes

    # ──────────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────────

    def save(self, directory, name):
        os.makedirs(directory, exist_ok=True)
        if self.zeven_freqs is None:
            raise RuntimeError("No results to save.")

        np.savez(
            os.path.join(directory, f"{name}.npz"),
            kpts=self.kpts, kpar=self.kpar,
            zeven_freqs=self.zeven_freqs, zodd_freqs=self.zodd_freqs,
        )
        meta = {
            'Lx_um': self.Lx, 'Ly_um': self.Ly,
            'thickness_um': self.thickness,
            'geometry_params_um': self.geometry_params,
            'geometry': self.geometry_type,
            'material_index': self.material_index,
            'num_bands': self.num_bands,
            'resolution': self.resolution,
            'factor_y': self.factor_y,
            'factor_z': self.factor_z,
            'zeven_gaps': [list(g) for g in self.zeven_gaps],
            'zodd_gaps': [list(g) for g in self.zodd_gaps],
        }
        with open(os.path.join(directory, f"{name}_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Saved: {directory}/{name}")

    def load(self, directory, name):
        data = np.load(os.path.join(directory, f"{name}.npz"))
        self.kpts = data['kpts']
        self.kpar = data['kpar']
        self.zeven_freqs = data['zeven_freqs']
        self.zodd_freqs = data['zodd_freqs']
        meta_path = os.path.join(directory, f"{name}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.zeven_gaps = [tuple(g) for g in meta.get('zeven_gaps', [])]
            self.zodd_gaps = [tuple(g) for g in meta.get('zodd_gaps', [])]
        print(f"Loaded: {directory}/{name}")