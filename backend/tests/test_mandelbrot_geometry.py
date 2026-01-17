"""Unit tests for Mandelbrot geometry implementation."""

import sys
import unittest
from pathlib import Path
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mandelbrot_orbits import (  # noqa: E402
    MandelbrotGeometry,
    LobeOrbitParams,
    MandelbrotOrbit,
    get_preset_orbit,
    list_preset_names,
)


class TestMandelbrotGeometry(unittest.TestCase):
    """Test Mandelbrot geometry functions."""

    def test_period2_exactness(self):
        """Test period-2 bulb has exact center and radius."""
        center = MandelbrotGeometry.period_n_bulb_center(2, 0)
        radius = MandelbrotGeometry.period_n_bulb_radius(2, 0)
        
        # Period-2 bulb should be at (-1, 0) with radius 0.25
        self.assertAlmostEqual(center.real, -1.0, places=10)
        self.assertAlmostEqual(center.imag, 0.0, places=10)
        self.assertAlmostEqual(radius, 0.25, places=10)

    def test_period3_hardcoded_centers(self):
        """Test period-3 bulb centers match exact values."""
        # sub_lobe 0: upper bulb
        center0 = MandelbrotGeometry.period_n_bulb_center(3, 0)
        self.assertAlmostEqual(center0.real, -0.122, places=3)
        self.assertAlmostEqual(center0.imag, 0.745, places=3)
        
        # sub_lobe 1: lower bulb
        center1 = MandelbrotGeometry.period_n_bulb_center(3, 1)
        self.assertAlmostEqual(center1.real, -0.122, places=3)
        self.assertAlmostEqual(center1.imag, -0.745, places=3)
        
        # sub_lobe 2: mini-Mandelbrot
        center2 = MandelbrotGeometry.period_n_bulb_center(3, 2)
        self.assertAlmostEqual(center2.real, -1.75, places=2)
        self.assertAlmostEqual(center2.imag, 0.0, places=3)

    def test_period4_cascade_tangency(self):
        """Test period-4 cascade bulb is tangent to period-2 circle."""
        # Get period-4 cascade center and radius
        r4 = MandelbrotGeometry.period_n_bulb_radius(4, 0)
        c4 = MandelbrotGeometry.period_n_bulb_center(4, 0)
        
        # Period-2 center at (-1, 0) with radius 0.25
        period2_center = complex(-1.0, 0.0)
        
        # Distance from c4 to period-2 center should be approximately (0.25 + r4)
        distance = abs(c4 - period2_center)
        expected_distance = 0.25 + r4
        
        self.assertAlmostEqual(distance, expected_distance, places=5,
                              msg=f"Period-4 cascade tangency: distance={distance}, expected={expected_distance}")

    def test_lobe_point_at_angle_bulb_boundary(self):
        """Test lobe_point_at_angle returns correct boundary point for bulbs."""
        # Period-2 bulb at theta=0, s=1.0 should give center + radius
        center = MandelbrotGeometry.period_n_bulb_center(2, 0)
        radius = MandelbrotGeometry.period_n_bulb_radius(2, 0)
        
        boundary_point = MandelbrotGeometry.lobe_point_at_angle(2, 0.0, 1.0, 0)
        expected = center + radius * complex(1, 0)
        
        self.assertAlmostEqual(boundary_point.real, expected.real, places=10)
        self.assertAlmostEqual(boundary_point.imag, expected.imag, places=10)

    def test_cardioid_function_consistency(self):
        """Test cardioid function gives expected value at theta=0."""
        # At theta=0: 0.5*e^(i*0) - 0.25*e^(i*0) = 0.5 - 0.25 = 0.25
        point = MandelbrotGeometry.main_cardioid_point(0.0)
        self.assertAlmostEqual(point.real, 0.25, places=10)
        self.assertAlmostEqual(point.imag, 0.0, places=10)

    def test_get_max_sub_lobe_for_period(self):
        """Test get_max_sub_lobe_for_period returns correct values."""
        self.assertEqual(MandelbrotGeometry.get_max_sub_lobe_for_period(1), 0)
        self.assertEqual(MandelbrotGeometry.get_max_sub_lobe_for_period(2), 0)
        self.assertEqual(MandelbrotGeometry.get_max_sub_lobe_for_period(3), 2)
        self.assertEqual(MandelbrotGeometry.get_max_sub_lobe_for_period(4), 2)
        self.assertEqual(MandelbrotGeometry.get_max_sub_lobe_for_period(8), 2)
        
        # For period-5, totient(5) = 4, so max_sub = 3
        self.assertEqual(MandelbrotGeometry.get_max_sub_lobe_for_period(5), 3)
        
        # For period-6, totient(6) = 2, so max_sub = 1
        self.assertEqual(MandelbrotGeometry.get_max_sub_lobe_for_period(6), 1)

    def test_euler_totient(self):
        """Test Euler totient function."""
        self.assertEqual(MandelbrotGeometry._euler_totient(1), 1)
        self.assertEqual(MandelbrotGeometry._euler_totient(2), 1)
        self.assertEqual(MandelbrotGeometry._euler_totient(3), 2)
        self.assertEqual(MandelbrotGeometry._euler_totient(4), 2)
        self.assertEqual(MandelbrotGeometry._euler_totient(5), 4)
        self.assertEqual(MandelbrotGeometry._euler_totient(6), 2)
        self.assertEqual(MandelbrotGeometry._euler_totient(7), 6)
        self.assertEqual(MandelbrotGeometry._euler_totient(8), 4)


class TestLobeOrbitParams(unittest.TestCase):
    """Test LobeOrbitParams dataclass."""

    def test_valid_params(self):
        """Test valid parameter creation."""
        params = LobeOrbitParams(lobe=3, sub_lobe=1, angular_velocity=1.5, s=1.02)
        self.assertEqual(params.lobe, 3)
        self.assertEqual(params.sub_lobe, 1)
        self.assertEqual(params.angular_velocity, 1.5)
        self.assertEqual(params.s, 1.02)

    def test_invalid_lobe(self):
        """Test that lobe < 1 raises error."""
        with self.assertRaises(ValueError):
            LobeOrbitParams(lobe=0, sub_lobe=0, angular_velocity=1.0, s=1.0)

    def test_invalid_sub_lobe(self):
        """Test that sub_lobe < 0 raises error."""
        with self.assertRaises(ValueError):
            LobeOrbitParams(lobe=2, sub_lobe=-1, angular_velocity=1.0, s=1.0)

    def test_invalid_s(self):
        """Test that s <= 0 raises error."""
        with self.assertRaises(ValueError):
            LobeOrbitParams(lobe=2, sub_lobe=0, angular_velocity=1.0, s=0.0)


class TestMandelbrotOrbit(unittest.TestCase):
    """Test MandelbrotOrbit class."""

    def test_orbit_initialization(self):
        """Test orbit can be initialized with new parameters."""
        orbit = MandelbrotOrbit(
            name="test_orbit",
            lobe=3,
            sub_lobe=1,
            n_points=50,
            s_range=(1.02, 1.02),
            angular_velocity=1.0
        )
        self.assertEqual(orbit.lobe, 3)
        self.assertEqual(orbit.sub_lobe, 1)
        self.assertIsNotNone(orbit.points)

    def test_orbit_sample(self):
        """Test orbit sampling returns correct shape."""
        orbit = MandelbrotOrbit(
            name="test",
            lobe=2,
            sub_lobe=0,
            n_points=50,
            s_range=(1.0, 1.0)
        )
        samples = orbit.sample(100)
        self.assertEqual(samples.shape, (100, 2))

    def test_orbit_velocities(self):
        """Test velocity computation returns correct shape."""
        orbit = MandelbrotOrbit(
            name="test",
            lobe=2,
            sub_lobe=0,
            n_points=50,
            s_range=(1.0, 1.0)
        )
        velocities = orbit.compute_velocities(100)
        self.assertEqual(velocities.shape, (100, 2))


class TestPresets(unittest.TestCase):
    """Test preset orbit functions."""

    def test_list_preset_names(self):
        """Test list_preset_names returns expected orbits."""
        names = list_preset_names()
        
        # Check for required presets
        self.assertIn("cardioid_boundary", names)
        self.assertIn("cardioid_interior", names)
        self.assertIn("period2_boundary", names)
        self.assertIn("period3_upper", names)
        self.assertIn("period3_lower", names)
        self.assertIn("period3_mini", names)
        self.assertIn("period4_cascade", names)
        self.assertIn("period4_primary_1", names)
        self.assertIn("period4_primary_2", names)
        self.assertIn("period8_cascade", names)
        self.assertIn("period8_primary_1", names)
        
        # Check for clockwise variants
        self.assertIn("cardioid_boundary_cw", names)
        self.assertIn("period2_boundary_cw", names)

    def test_get_preset_orbit(self):
        """Test get_preset_orbit returns valid orbits."""
        orbit = get_preset_orbit("cardioid_boundary")
        self.assertEqual(orbit.lobe, 1)
        self.assertEqual(orbit.sub_lobe, 0)
        
        # Test period-3 presets
        orbit_upper = get_preset_orbit("period3_upper")
        self.assertEqual(orbit_upper.lobe, 3)
        self.assertEqual(orbit_upper.sub_lobe, 0)
        
        orbit_lower = get_preset_orbit("period3_lower")
        self.assertEqual(orbit_lower.lobe, 3)
        self.assertEqual(orbit_lower.sub_lobe, 1)
        
        orbit_mini = get_preset_orbit("period3_mini")
        self.assertEqual(orbit_mini.lobe, 3)
        self.assertEqual(orbit_mini.sub_lobe, 2)

    def test_period4_presets(self):
        """Test period-4 presets are correctly configured."""
        cascade = get_preset_orbit("period4_cascade")
        self.assertEqual(cascade.lobe, 4)
        self.assertEqual(cascade.sub_lobe, 0)
        
        primary1 = get_preset_orbit("period4_primary_1")
        self.assertEqual(primary1.lobe, 4)
        self.assertEqual(primary1.sub_lobe, 1)
        
        primary2 = get_preset_orbit("period4_primary_2")
        self.assertEqual(primary2.lobe, 4)
        self.assertEqual(primary2.sub_lobe, 2)

    def test_backward_compatibility(self):
        """Test backward compatible preset names still work."""
        # These should still exist for compatibility
        orbit = get_preset_orbit("period3_boundary")
        self.assertIsNotNone(orbit)
        
        orbit = get_preset_orbit("period4_boundary")
        self.assertIsNotNone(orbit)


if __name__ == "__main__":
    unittest.main()
