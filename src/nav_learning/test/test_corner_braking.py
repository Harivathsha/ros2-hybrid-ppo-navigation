"""Check the added reward without ROS, PyTorch, or Gazebo."""
import math
import unittest
from dataclasses import replace

from nav_learning.nav_math import RewardSettings, calculate_reward


class CornerBrakingTests(unittest.TestCase):
    def reward(self, speed=.22, previous=.42, current=.40, scale=1.0):
        return calculate_reward(
            previous_goal_distance_m=2.0,
            current_goal_distance_m=1.98,
            relative_goal_bearing_rad=0.0,
            nearest_obstacle_m=current,
            commanded_linear_velocity_mps=speed,
            outcome="running",
            reward_settings=RewardSettings(
                safety_distance_m=.50, near_obstacle_penalty_scale=.20,
                braking_penalty_scale=scale),
            previous_nearest_obstacle_m=previous,
        )

    def test_fast_approach_gets_an_added_cost(self):
        self.assertLess(self.reward()[1]["braking_penalty"], 0)

    def test_slower_approach_gets_a_smaller_added_cost(self):
        fast = self.reward(speed=.22, previous=.42)[1]["braking_penalty"]
        slow = self.reward(speed=.11, previous=.41)[1]["braking_penalty"]
        self.assertLess(fast, slow)
        self.assertLess(slow, 0)

    def test_turning_in_place_gets_no_added_cost(self):
        self.assertEqual(self.reward(speed=0)[1]["braking_penalty"], 0)

    def test_opening_clearance_gets_no_added_cost(self):
        self.assertEqual(self.reward(previous=.38)[1]["braking_penalty"], 0)

    def test_parallel_motion_gets_no_added_cost(self):
        self.assertEqual(self.reward(previous=.40)[1]["braking_penalty"], 0)

    def test_small_range_changes_are_in_the_noise_deadband(self):
        self.assertEqual(self.reward(previous=.401)[1]["braking_penalty"], 0)

    def test_clear_space_gets_no_added_cost(self):
        self.assertEqual(self.reward(previous=1.02, current=1.0)[1]["braking_penalty"], 0)

    def test_old_checkpoints_default_to_disabled(self):
        settings = RewardSettings.from_mapping({})
        self.assertEqual(settings.braking_penalty_scale, 0)
        reward, parts = calculate_reward(2, 1.98, 0, .4, .22, "running")
        self.assertTrue(math.isfinite(reward))
        self.assertEqual(parts["braking_penalty"], 0)

    def test_disabling_preserves_the_legacy_reward(self):
        a = self.reward(previous=.42, scale=0)
        b = self.reward(previous=None, scale=0)
        self.assertEqual(a, b)

    def test_gain_scales_only_the_added_cost(self):
        a = self.reward(scale=1)[1]
        b = self.reward(scale=2)[1]
        self.assertAlmostEqual(b["braking_penalty"], 2*a["braking_penalty"])
        self.assertEqual(a["obstacle_penalty"], b["obstacle_penalty"])
        self.assertEqual(a["progress_reward"], b["progress_reward"])

    def test_enabled_reward_requires_previous_measurement(self):
        for previous in (None, -1, float("nan"), float("inf")):
            with self.subTest(previous=previous), self.assertRaises(ValueError):
                self.reward(previous=previous)

    def test_bad_gains_are_rejected(self):
        for gain in (-1, True, float("nan"), float("inf")):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                RewardSettings.from_mapping({"braking_penalty_scale": gain})

    def test_cost_is_finite_and_bounded(self):
        for speed in (0, .01, .1, .22):
            for previous in (.15, .22, .3, .5, 8.0):
                for current in (.15, .22, .3, .5, 8.0):
                    p = self.reward(speed, previous, current)[1]["braking_penalty"]
                    self.assertTrue(math.isfinite(p))
                    self.assertGreaterEqual(p, -1)
                    self.assertLessEqual(p, 0)


if __name__ == "__main__":
    unittest.main()
