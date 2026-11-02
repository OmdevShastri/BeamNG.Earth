"""server/interchange.py: Unified height optimization for overpasses, underpasses, and interchanges (GitHub #627).

Instead of heuristic per-road rules (_lower_underpasses / _lift_approaches), an interchange or grade-separated
cluster is solved as a single constrained quadratic program (QP).

Calls the Rust kernel in earthfast.interchange_solve when available; falls back to the pure-NumPy ADMM solver.
"""

import math
import numpy as np

# Road class stiffness weights: how much a road resists vertical movement
# High weights = stiff (anchored to surroundings, like urban street grids & frontage roads)
# Low weights = flexible (designed to fly or dive, like connectors and ramps)
CLASS_WEIGHTS = {
    "motorway": 8.0,
    "trunk": 8.0,
    "primary": 6.0,
    "secondary": 4.0,
    "tertiary": 3.0,
    "residential": 12.0,
    "living_street": 15.0,
    "service": 15.0,
    "unclassified": 5.0,
    "motorway_link": 1.0,
    "trunk_link": 1.0,
    "primary_link": 1.5,
    "secondary_link": 2.0,
    "tertiary_link": 2.0,
}

# Maximum grades per road class (rise over run)
CLASS_GMAX = {
    "motorway": 0.035,
    "trunk": 0.040,
    "primary": 0.050,
    "secondary": 0.060,
    "tertiary": 0.070,
    "residential": 0.100,
    "living_street": 0.120,
    "service": 0.120,
    "unclassified": 0.080,
    "motorway_link": 0.060,
    "trunk_link": 0.060,
    "primary_link": 0.070,
    "secondary_link": 0.070,
    "tertiary_link": 0.070,
}

DEFAULT_WEIGHT = 4.0
DEFAULT_GMAX = 0.06
DEFAULT_CLEARANCE = 6.5  # 5.0m headroom + 1.5m deck depth


def class_weight(cls):
    """Stiffness weight for road class string."""
    return CLASS_WEIGHTS.get(cls, DEFAULT_WEIGHT)


def class_gmax(cls):
    """Maximum longitudinal grade for road class string."""
    return CLASS_GMAX.get(cls, DEFAULT_GMAX)


def solve_admm_numpy(z_dem, off, step, gmax, weight, smooth_weight, fixed_ends, junctions, crossings,
                     max_iter=600, eps_abs=1e-4, ds=None, z_min=None):
    """Pure-NumPy ADMM fallback for the QP solver."""
    n = len(z_dem)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    nr = len(off) - 1

    # 1. Build P and q
    P = np.zeros((n, n), dtype=np.float64)
    q = np.zeros(n, dtype=np.float64)

    for r in range(nr):
        r0, r1 = off[r], off[r + 1]
        w = weight[r]
        for i in range(r0, r1):
            P[i, i] += w
            q[i] -= w * z_dem[i]

        if r1 - r0 >= 3 and smooth_weight > 0.0:
            sw = smooth_weight
            for i in range(r0 + 1, r1 - 1):
                im1, ip1 = i - 1, i + 1
                P[im1, im1] += sw
                P[im1, i]   -= 2.0 * sw
                P[im1, ip1] += sw
                P[i, im1]   -= 2.0 * sw
                P[i, i]     += 4.0 * sw
                P[i, ip1]   -= 2.0 * sw
                P[ip1, im1] += sw
                P[ip1, i]   -= 2.0 * sw
                P[ip1, ip1] += sw

    # 2. Build constraints
    A_rows, l_rows, u_rows = [], [], []

    seg_idx = 0
    for r in range(nr):
        r0, r1 = off[r], off[r + 1]
        for i in range(r0, r1 - 1):
            if ds is not None and seg_idx < len(ds):
                max_dz = gmax[r] * float(ds[seg_idx])
            else:
                max_dz = gmax[r] * step[r]
            seg_idx += 1
            row = np.zeros(n)
            row[i + 1] = 1.0
            row[i] = -1.0
            A_rows.append(row)
            l_rows.append(-max_dz)
            u_rows.append(max_dz)

        if fixed_ends[r, 0]:
            row = np.zeros(n)
            row[r0] = 1.0
            A_rows.append(row)
            l_rows.append(z_dem[r0])
            u_rows.append(z_dem[r0])

        if fixed_ends[r, 1]:
            last = r1 - 1
            row = np.zeros(n)
            row[last] = 1.0
            A_rows.append(row)
            l_rows.append(z_dem[last])
            u_rows.append(z_dem[last])

    for ia, ib in junctions:
        if ia < n and ib < n:
            row = np.zeros(n)
            row[ia] = 1.0
            row[ib] = -1.0
            A_rows.append(row)
            l_rows.append(0.0)
            u_rows.append(0.0)

    # Lower bound / floor constraints (sea floor, polder protection)
    if z_min is not None:
        for i in range(min(n, len(z_min))):
            if z_min[i] > -1e8:
                row = np.zeros(n)
                row[i] = 1.0
                A_rows.append(row)
                l_rows.append(float(z_min[i]))
                u_rows.append(np.inf)

    for c_spec in crossings:
        if len(c_spec) == 3:
            iu, il, cl = c_spec
            iu, il = int(iu), int(il)
            if iu < n and il < n:
                row = np.zeros(n)
                row[iu] = 1.0
                row[il] = -1.0
                A_rows.append(row)
                l_rows.append(cl)
                u_rows.append(np.inf)
        elif len(c_spec) == 6:
            iu0, w0, iu1, w1, il, cl = c_spec
            iu0, iu1, il = int(iu0), int(iu1), int(il)
            if iu0 < n and iu1 < n and il < n:
                row = np.zeros(n)
                row[iu0] += float(w0)
                row[iu1] += float(w1)
                row[il] -= 1.0
                A_rows.append(row)
                l_rows.append(cl)
                u_rows.append(np.inf)

    if not A_rows:
        return z_dem.copy()

    A = np.vstack(A_rows)
    l = np.array(l_rows, dtype=np.float64)
    u = np.array(u_rows, dtype=np.float64)

    m = len(l)
    rho = 15.0
    sigma = 1e-4

    AT = A.T
    M = P + sigma * np.eye(n) + rho * (AT @ A)
    M_inv = np.linalg.inv(M)

    z = z_dem.copy()
    x = np.zeros(m, dtype=np.float64)
    y = np.zeros(m, dtype=np.float64)

    for it in range(max_iter):
        rhs = sigma * z - q + AT @ (rho * x - y)
        z_next = M_inv @ rhs

        Az = A @ z_next
        x_next = np.clip(Az + y / rho, l, u)
        y = y + rho * (Az - x_next)

        if np.max(np.abs(Az - x_next)) < eps_abs and np.max(np.abs(z_next - z)) < eps_abs:
            break
        z = z_next
        x = x_next

    return z_next


def solve_interchange(z_dem, off, step, gmax, weight, smooth_weight=40.0, fixed_ends=None, junctions=None,
                      crossings=None, ds=None, z_min=None):
    """Solves the unified interchange height problem across multiple interacting roads.

    Returns the solved elevation array z of shape (N,).
    """
    nr = len(off) - 1
    if fixed_ends is None:
        fixed_ends = np.ones((nr, 2), dtype=np.uint8)
    if junctions is None:
        junctions = np.zeros((0, 2), dtype=np.int64)
    if crossings is None:
        crossings = np.zeros((0, 3), dtype=np.float64)

    import fast
    if hasattr(fast, "_ef") and hasattr(fast._ef, "interchange_solve"):
        try:
            return np.asarray(
                fast._ef.interchange_solve(
                    np.ascontiguousarray(z_dem, dtype=np.float64),
                    np.ascontiguousarray(off, dtype=np.int64),
                    np.ascontiguousarray(step, dtype=np.float64),
                    np.ascontiguousarray(gmax, dtype=np.float64),
                    np.ascontiguousarray(weight, dtype=np.float64),
                    float(smooth_weight),
                    np.ascontiguousarray(fixed_ends, dtype=np.uint8),
                    np.ascontiguousarray(junctions, dtype=np.int64),
                    np.ascontiguousarray(crossings, dtype=np.float64),
                    600,
                    1e-4,
                ),
                dtype=np.float64,
            )
        except Exception:
            pass

    return solve_admm_numpy(
        np.asarray(z_dem, dtype=np.float64),
        np.asarray(off, dtype=np.int64),
        np.asarray(step, dtype=np.float64),
        np.asarray(gmax, dtype=np.float64),
        np.asarray(weight, dtype=np.float64),
        float(smooth_weight),
        np.asarray(fixed_ends, dtype=np.uint8),
        np.asarray(junctions, dtype=np.int64),
        np.asarray(crossings, dtype=np.float64),
        ds=None if ds is None else np.asarray(ds, dtype=np.float64),
        z_min=None if z_min is None else np.asarray(z_min, dtype=np.float64),
    )

