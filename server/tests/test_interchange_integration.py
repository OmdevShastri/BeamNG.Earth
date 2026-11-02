"""Integration test for unified interchange grading (INTERCHANGE_ON, GitHub #627)."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import grading
import roads
from test_roads import features, FRAME
from test_grading import FakeMD

FLAT = 10.0


def flat_elev(la, lo):
    return np.full(np.shape(la), FLAT)


def overpass():
    """A secondary road east-west over x = -400..400, a bridge over x = -35..35, crossing a divided motorway (two
    carriageways north-south at x = -12 and x = +12)."""
    a, b = (400 - 35) / 800, (400 + 35) / 800
    return features(
        [[(-400, 0), (400, 0)], [(-12, -500), (-12, 500)], [(12, 500), (12, -500)]],
        ["secondary", "motorway", "motorway"],
        flags=[(0, "is_bridge", a, b)],
        ids=["cross", "i20n", "i20s"],
    )


class InterchangeIntegrationTest(unittest.TestCase):
    def test_interchange_solver_clears_crossing(self):
        old_ic = grading.INTERCHANGE_ON
        grading.INTERCHANGE_ON = True
        try:
            f = overpass()
            g = grading.Grader(FRAME, FakeMD(f), flat_elev)

            def elev(la, lo):
                return g.apply(la, lo, flat_elev(la, lo), grading.ROAD_LIFT)

            bp = roads.BridgeProfiles(FRAME, lambda key: f, elev)
            parts = bp.parts([f], (-100, -100, 100, 100))
            self.assertEqual(len(parts), 1)

            _fid, _spec, xy, z, _lon = parts[0]
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.asarray(xy)[:, :2], axis=0).T))]
            at = [float(np.interp(x, np.asarray(xy)[:, 0], s)) for x in (-12.0, 12.0)]
            lanes = elev(*FRAME.enu_to_geodetic(np.array([-12.0, 12.0]), np.array([0.0, 0.0]), 0.0)[:2])

            clearance = np.interp(at, s, z) - lanes
            depth = roads.deck_depth(roads.geom_span([np.c_[np.linspace(-35, 35, 36), np.zeros(36)]]))

            # Deck must clear the lanes
            self.assertGreaterEqual(clearance.min(), roads.deck_road_clear(depth) - 0.1)

            # Headroom under girders >= 4.9m (16 ft)
            under = clearance.min() - depth
            self.assertGreater(under, 4.8)
        finally:
            grading.INTERCHANGE_ON = old_ic

    def test_motorway_overpass_wezep_pattern(self):
        """A motorway overpass (A28 style) over a local road (Zuiderzeestraatweg).
        The motorway overpass must rise to give clearance, overcoming legacy APPROACH_SKIP."""
        old_ic = grading.INTERCHANGE_ON
        grading.INTERCHANGE_ON = True
        try:
            # Motorway north-south over y = -500..500 with bridge over y = -20..20,
            # crossing an east-west secondary road at y = 0.
            a, b = (500 - 20) / 1000, (500 + 20) / 1000
            f = features(
                [[(0, -500), (0, 500)], [(-400, 0), (400, 0)]],
                ["motorway", "secondary"],
                flags=[(0, "is_bridge", a, b)],
                ids=["a28", "local"],
            )
            g = grading.Grader(FRAME, FakeMD(f), flat_elev)

            def elev(la, lo):
                return g.apply(la, lo, flat_elev(la, lo), grading.ROAD_LIFT)

            bp = roads.BridgeProfiles(FRAME, lambda key: f, elev)
            parts = bp.parts([f], (-50, -50, 50, 50))
            self.assertEqual(len(parts), 1)

            _fid, _spec, xy, z, _lon = parts[0]
            # Center of the bridge deck at (0, 0)
            deck_center_z = float(np.interp(0.0, np.asarray(xy)[:, 1], np.asarray(z)))
            under_z = elev(*FRAME.enu_to_geodetic(0.0, 0.0, 0.0)[:2])
            clearance = deck_center_z - under_z

            span_len = 40.0
            depth = roads.deck_depth(span_len)
            req_clear = roads.deck_road_clear(depth)

            # Motorway bridge deck must clear the secondary street below (headroom >= 4.9m / 16 ft)
            self.assertGreater(clearance, 7.0)
            under = clearance - depth
            self.assertGreater(under, 4.8)
        finally:
            grading.INTERCHANGE_ON = old_ic


if __name__ == "__main__":
    unittest.main()

