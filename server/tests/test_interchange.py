"""Unit tests for server/interchange.py (GitHub #627: Unified Interchange Height Solver)."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import interchange


class InterchangeSolverTest(unittest.TestCase):
    def test_single_overpass_clearance(self):
        # Road 0: Secondary overpass (600 m, 61 points, ds=10 m, gmax=0.06, weight=4.0)
        # Road 1: Motorway underpass (600 m, 61 points, ds=10 m, gmax=0.035, weight=8.0)
        # Both start at flat DEM = 10.0 m
        n0 = 61
        n1 = 61
        z_dem = np.full(n0 + n1, 10.0, dtype=np.float64)
        off = np.array([0, n0, n0 + n1], dtype=np.int64)
        step = np.array([10.0, 10.0], dtype=np.float64)
        gmax = np.array([0.060, 0.035], dtype=np.float64)
        weight = np.array([4.0, 8.0], dtype=np.float64)

        # Crossing at the middle of both roads (s = 300 m => idx 30)
        crossings = np.array([[30, n0 + 30, 6.5]], dtype=np.float64)

        z_sol = interchange.solve_interchange(
            z_dem, off, step, gmax, weight, smooth_weight=40.0, crossings=crossings
        )

        z0 = z_sol[:n0]
        z1 = z_sol[n0:]

        # 1. Check crossing clearance
        clearance = z0[30] - z1[30]
        self.assertGreaterEqual(clearance, 6.49)

        # 2. Check that the stiffer motorway moved less than the flexible secondary road
        rise_upper = z0[30] - 10.0
        dip_lower = 10.0 - z1[30]
        self.assertGreater(rise_upper, dip_lower)

        # 3. Check grade constraints along both roads
        g0 = np.max(np.abs(np.diff(z0) / 10.0))
        g1 = np.max(np.abs(np.diff(z1) / 10.0))
        self.assertLessEqual(g0, 0.060 + 1e-4)
        self.assertLessEqual(g1, 0.035 + 1e-4)

    def test_ramp_junction_connection(self):
        # Road 0: Motorway (800 m, 81 pts, ds=10 m)
        # Road 1: Overpass (600 m, 61 pts, ds=10 m)
        # Road 2: Off-ramp (300 m, 31 pts, ds=10 m, weight=1.0, gmax=0.06)
        n0, n1, n2 = 81, 61, 31
        total = n0 + n1 + n2
        z_dem = np.full(total, 10.0, dtype=np.float64)
        off = np.array([0, n0, n0 + n1, total], dtype=np.int64)
        step = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        gmax = np.array([0.035, 0.050, 0.060], dtype=np.float64)
        weight = np.array([10.0, 3.0, 1.0], dtype=np.float64)

        # Overpass crosses over motorway at center
        crossings = np.array([[n0 + 30, 40, 6.5]], dtype=np.float64)

        # Ramp junctions:
        # starts at Motorway s=200 m (idx 20)
        # ends at Overpass approach s=260 m (idx n0 + 26)
        ramp_start = n0 + n1
        ramp_end = total - 1
        junctions = np.array([[ramp_start, 20], [ramp_end, n0 + 26]], dtype=np.int64)

        # Ramp boundary is not fixed to ground; only mainline ends are
        fixed_ends = np.array([[1, 1], [1, 1], [0, 0]], dtype=np.uint8)

        z_sol = interchange.solve_interchange(
            z_dem, off, step, gmax, weight, fixed_ends=fixed_ends, junctions=junctions, crossings=crossings
        )

        z_fwy = z_sol[:n0]
        z_art = z_sol[n0 : n0 + n1]
        z_ramp = z_sol[n0 + n1 :]

        # Verify exact junction matching
        self.assertAlmostEqual(z_ramp[0], z_fwy[20], places=3)
        self.assertAlmostEqual(z_ramp[-1], z_art[26], places=3)

        # Verify ramp climbed smoothly to meet the bridge
        self.assertGreater(z_ramp[-1], z_ramp[0] + 1.0)

        # Verify ramp slope limit
        g_ramp = np.max(np.abs(np.diff(z_ramp) / 10.0))
        self.assertLessEqual(g_ramp, 0.060 + 1e-4)


if __name__ == "__main__":
    unittest.main()

