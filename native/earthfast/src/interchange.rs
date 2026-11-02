//! interchange.rs: Unified height optimization for overpasses, underpasses, and interchanges (GitHub #627).
//!
//! Replaces heuristic per-road rules (_lower_underpasses, _lift_approaches, UNDER_LIDAR) with a single
//! constrained quadratic program (ADMM solver).
//! Pure, deterministic, zero external dependencies.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// A single linear constraint: l <= a1 * z[i1] (+ a2 * z[i2]) (+ a3 * z[i3]) <= u
#[derive(Clone, Copy)]
struct Constraint {
    i1: usize,
    a1: f64,
    i2: Option<(usize, f64)>,
    i3: Option<(usize, f64)>,
    l: f64,
    u: f64,
}

/// Dense Cholesky decomposition of symmetric positive definite matrix A (n x n).
/// Returns lower-triangular matrix L such that A = L * L^T.
fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let val = a[i * n + i] - sum;
                if val <= 0.0 {
                    return None; // Not positive definite
                }
                l[i * n + j] = val.sqrt();
            } else {
                let diag = l[j * n + j];
                if diag.abs() < 1e-12 {
                    return None;
                }
                l[i * n + j] = (a[i * n + j] - sum) / diag;
            }
        }
    }
    Some(l)
}

/// Solves (L * L^T) x = b using forward and back substitution.
fn cholesky_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    // Forward substitution: L y = b
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..i {
            sum += l[i * n + k] * y[k];
        }
        y[i] = (b[i] - sum) / l[i * n + i];
    }
    // Backward substitution: L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for k in (i + 1)..n {
            sum += l[k * n + i] * x[k];
        }
        x[i] = (y[i] - sum) / l[i * n + i];
    }
    x
}

/// Solves the unified interchange height optimization problem.
///
/// Inputs:
/// - z_dem: (n,) target DEM elevations across all roads concatenated
/// - off: (nr + 1,) point offset array for each road
/// - step: (nr,) station distance step in meters for each road
/// - gmax: (nr,) maximum grade limit for each road (e.g. 0.035 for motorways, 0.06 for ramps)
/// - weight: (nr,) earthwork stiffness weight for each road
/// - smooth_weight: curvature / acceleration smoothing factor (lambda)
/// - fixed_ends: (nr, 2) boolean flags indicating whether road start / end is fixed to z_dem
/// - junctions: (nj, 2) pairs of global vertex indices [idx_a, idx_b] required to meet at grade (zA == zB)
/// - crossings: (nc, 3) grade separations [upper_idx, lower_idx, min_clearance]
/// - max_iter: maximum ADMM iterations (e.g. 500)
/// - eps_abs: absolute convergence tolerance (e.g. 1e-4)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn interchange_solve<'py>(
    py: Python<'py>,
    z_dem: PyReadonlyArray1<'py, f64>,
    off: PyReadonlyArray1<'py, i64>,
    step: PyReadonlyArray1<'py, f64>,
    gmax: PyReadonlyArray1<'py, f64>,
    weight: PyReadonlyArray1<'py, f64>,
    smooth_weight: f64,
    fixed_ends: PyReadonlyArray2<'py, u8>,
    junctions: PyReadonlyArray2<'py, i64>,
    crossings: PyReadonlyArray2<'py, f64>,
    max_iter: usize,
    eps_abs: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let z_dem = z_dem.as_array();
    let n = z_dem.len();
    if n == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let off_arr = off.as_array();
    let nr = off_arr.len().saturating_sub(1);
    let off: Vec<usize> = off_arr.iter().map(|&v| v.max(0) as usize).collect();

    let step = step.as_array();
    let gmax = gmax.as_array();
    let weight = weight.as_array();
    let fixed_ends = fixed_ends.as_array();

    // 1. Build P matrix (n x n) and q vector (n)
    // Objective: (1/2) z^T P z + q^T z
    let mut p = vec![0.0; n * n];
    let mut q = vec![0.0; n];

    for r in 0..nr {
        let (r0, r1) = (off[r], off[r + 1]);
        if r1 <= r0 {
            continue;
        }
        let w = weight[r];

        // Ground tracking: w * (z_i - dem_i)^2 => P_ii += w, q_i -= w * dem_i
        for i in r0..r1 {
            p[i * n + i] += w;
            q[i] -= w * z_dem[i];
        }

        // Curvature / Smoothness: lambda * (z_{i-1} - 2*z_i + z_{i+1})^2
        if r1 - r0 >= 3 && smooth_weight > 0.0 {
            for i in (r0 + 1)..(r1 - 1) {
                let im1 = i - 1;
                let ip1 = i + 1;

                p[im1 * n + im1] += smooth_weight * 1.0;
                p[i * n + i] += smooth_weight * 4.0;
                p[ip1 * n + ip1] += smooth_weight * 1.0;

                p[im1 * n + i] -= smooth_weight * 2.0;
                p[i * n + im1] -= smooth_weight * 2.0;

                p[i * n + ip1] -= smooth_weight * 2.0;
                p[ip1 * n + i] -= smooth_weight * 2.0;

                p[im1 * n + ip1] += smooth_weight * 1.0;
                p[ip1 * n + im1] += smooth_weight * 1.0;
            }
        }
    }

    // 2. Assemble constraints: l <= A z <= u
    let mut constraints: Vec<Constraint> = Vec::new();

    // A. Grade limits along each road: -gmax * ds <= z_{i+1} - z_i <= gmax * ds
    for r in 0..nr {
        let (r0, r1) = (off[r], off[r + 1]);
        if r1 <= r0 + 1 {
            continue;
        }
        let max_dz = gmax[r] * step[r];
        for i in r0..(r1 - 1) {
            constraints.push(Constraint {
                i1: i + 1,
                a1: 1.0,
                i2: Some((i, -1.0)),
                i3: None,
                l: -max_dz,
                u: max_dz,
            });
        }

        // B. Fixed boundary ends
        if fixed_ends[[r, 0]] != 0 {
            constraints.push(Constraint {
                i1: r0,
                a1: 1.0,
                i2: None,
                i3: None,
                l: z_dem[r0],
                u: z_dem[r0],
            });
        }
        if fixed_ends[[r, 1]] != 0 {
            let last = r1 - 1;
            constraints.push(Constraint {
                i1: last,
                a1: 1.0,
                i2: None,
                i3: None,
                l: z_dem[last],
                u: z_dem[last],
            });
        }
    }

    // C. At-grade junctions: z[idx_a] - z[idx_b] == 0
    let junc = junctions.as_array();
    for j in 0..junc.nrows() {
        let ia = junc[[j, 0]] as usize;
        let ib = junc[[j, 1]] as usize;
        if ia < n && ib < n {
            constraints.push(Constraint {
                i1: ia,
                a1: 1.0,
                i2: Some((ib, -1.0)),
                i3: None,
                l: 0.0,
                u: 0.0,
            });
        }
    }

    // D. Grade-separated crossings: z[upper] - z[lower] >= clearance (or 2-point bridge deck)
    let cross = crossings.as_array();
    let cross_cols = cross.ncols();
    for c in 0..cross.nrows() {
        if cross_cols == 3 {
            let iu = cross[[c, 0]] as usize;
            let il = cross[[c, 1]] as usize;
            let clear = cross[[c, 2]];
            if iu < n && il < n {
                constraints.push(Constraint {
                    i1: iu,
                    a1: 1.0,
                    i2: Some((il, -1.0)),
                    i3: None,
                    l: clear,
                    u: f64::INFINITY,
                });
            }
        } else if cross_cols >= 6 {
            let iu0 = cross[[c, 0]] as usize;
            let w0 = cross[[c, 1]];
            let iu1 = cross[[c, 2]] as usize;
            let w1 = cross[[c, 3]];
            let il = cross[[c, 4]] as usize;
            let clear = cross[[c, 5]];
            if iu0 < n && iu1 < n && il < n {
                if iu0 == iu1 {
                    constraints.push(Constraint {
                        i1: iu0,
                        a1: w0 + w1,
                        i2: Some((il, -1.0)),
                        i3: None,
                        l: clear,
                        u: f64::INFINITY,
                    });
                } else {
                    constraints.push(Constraint {
                        i1: iu0,
                        a1: w0,
                        i2: Some((iu1, w1)),
                        i3: Some((il, -1.0)),
                        l: clear,
                        u: f64::INFINITY,
                    });
                }
            }
        }
    }

    let m_constraints = constraints.len();
    let rho = 15.0;
    let sigma = 1e-4;

    // 3. Form system matrix: M = P + sigma * I + rho * A^T A
    let mut m_mat = p;
    for i in 0..n {
        m_mat[i * n + i] += sigma;
    }

    // Add rho * A^T A
    for c in &constraints {
        let mut terms = [(c.i1, c.a1), (0, 0.0), (0, 0.0)];
        let mut k = 1;
        if let Some(t2) = c.i2 {
            terms[k] = t2;
            k += 1;
        }
        if let Some(t3) = c.i3 {
            terms[k] = t3;
            k += 1;
        }
        for a in 0..k {
            let (ia, aa) = terms[a];
            for b in 0..k {
                let (ib, ab) = terms[b];
                m_mat[ia * n + ib] += rho * aa * ab;
            }
        }
    }

    // Cholesky factorize M
    let l_chol = match cholesky(&m_mat, n) {
        Some(l) => l,
        None => {
            // Fallback: return input DEM if matrix is ill-conditioned
            return Ok(z_dem.to_vec().into_pyarray(py));
        }
    };

    // 4. ADMM Iteration
    let mut z = z_dem.to_vec();
    let mut x = vec![0.0; m_constraints];
    let mut y = vec![0.0; m_constraints];

    // Initialize x = A * z
    for (k, c) in constraints.iter().enumerate() {
        let mut az = c.a1 * z[c.i1];
        if let Some((i2, a2)) = c.i2 {
            az += a2 * z[i2];
        }
        if let Some((i3, a3)) = c.i3 {
            az += a3 * z[i3];
        }
        x[k] = az.clamp(c.l, c.u);
    }

    let mut rhs = vec![0.0; n];

    for _iter in 0..max_iter {
        // rhs = sigma * z - q + A^T (rho * x - y)
        for i in 0..n {
            rhs[i] = sigma * z[i] - q[i];
        }
        for (k, c) in constraints.iter().enumerate() {
            let v = rho * x[k] - y[k];
            rhs[c.i1] += c.a1 * v;
            if let Some((i2, a2)) = c.i2 {
                rhs[i2] += a2 * v;
            }
            if let Some((i3, a3)) = c.i3 {
                rhs[i3] += a3 * v;
            }
        }

        // Solve M * z_next = rhs
        let z_next = cholesky_solve(&l_chol, n, &rhs);

        // x_next = clamp(A * z_next + y / rho, l, u)
        let mut max_prim_res = 0.0f64;
        let mut max_step_res = 0.0f64;

        for (k, c) in constraints.iter().enumerate() {
            let mut az_next = c.a1 * z_next[c.i1];
            if let Some((i2, a2)) = c.i2 {
                az_next += a2 * z_next[i2];
            }
            if let Some((i3, a3)) = c.i3 {
                az_next += a3 * z_next[i3];
            }

            let x_next = (az_next + y[k] / rho).clamp(c.l, c.u);
            y[k] += rho * (az_next - x_next);

            let prim_res = (az_next - x_next).abs();
            if prim_res > max_prim_res {
                max_prim_res = prim_res;
            }

            x[k] = x_next;
        }

        for i in 0..n {
            let step_res = (z_next[i] - z[i]).abs();
            if step_res > max_step_res {
                max_step_res = step_res;
            }
            z[i] = z_next[i];
        }

        if max_prim_res < eps_abs && max_step_res < eps_abs {
            break;
        }
    }

    Ok(z.into_pyarray(py))
}

