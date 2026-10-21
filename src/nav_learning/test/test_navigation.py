"""Offline tests. These validate geometry and math, not Gazebo physics or learning."""
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from nav_learning.nav_math import (
    ARENA_HALF_M, BOX_COUNT, BOX_HEIGHT_M, BOX_NAMES, BOX_SIZE_M,
    CURRICULUM_DISTANCE_LIMITS_M, ENVIRONMENT_ID, GOAL_SUCCESS_RADIUS_M, GOAL_RADIUS_M,
    STAGES, GOAL_CLEARANCE_M, parked_box_pose, straight_path_blocker_count,
    MIN_BOX_GAP_M, MIN_BOX_WALL_GAP_M, WORLD_NAME, ensure_checkpoint_environment,
    goal_region_reached, navigation_path_exists, point_is_clear,
    sample_episode, straight_path_is_clear,
)
from nav_learning.nav_math import (
    COLLISION_DISTANCE_M, MAX_LINEAR_SPEED_MPS, OBSERVATION_SIZE, InvalidScanError,
    build_observation, calculate_reward, determine_episode_outcome,
    policy_action_to_velocity, preprocess_scan, relative_goal, velocity_to_policy_action,
)


class ArenaContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = {(level, seed): sample_episode(seed, level)
                        for level in (0, 1, 2) for seed in range(10000, 10030)}

    def test_seed_repeatability(self):
        for level in (0, 1, 2):
            self.assertEqual(self.episodes[level,10000], sample_episode(10000,level))

    def test_stages_have_random_endpoints_and_active_counts(self):
        for level in (0,1,2):
            a, b = self.episodes[level,10000], self.episodes[level,10001]
            low, high = STAGES[level].box_count_limits
            self.assertTrue(low <= len(a.layout) <= high)
            for name in set(a.layout) & set(b.layout):
                self.assertNotEqual(a.layout[name][:2], b.layout[name][:2])
            self.assertNotEqual(a.start, b.start)
            self.assertNotEqual(a.goal, b.goal)

    def test_every_spawn_and_box_fits_its_patch(self):
        for episode in self.episodes.values():
            half = episode.region_size_m / 2
            cx, cy = episode.region_center
            for point in (episode.start[:2], episode.goal):
                self.assertLessEqual(max(abs(point[0] - cx), abs(point[1] - cy)) + .35, half + 1e-9)
            for pose in episode.layout.values():
                self.assertLessEqual(max(abs(pose[0] - cx), abs(pose[1] - cy)) + BOX_SIZE_M / 2, half + 1e-9)

    def test_no_overlaps_and_wall_clearance(self):
        for episode in self.episodes.values():
            poses = list(episode.layout.values())
            for i, p in enumerate(poses):
                self.assertLessEqual(max(abs(p[0]),abs(p[1]))+BOX_SIZE_M/2+MIN_BOX_WALL_GAP_M, ARENA_HALF_M+1e-9)
                self.assertEqual(p[2], BOX_HEIGHT_M/2)
                self.assertEqual(p[3], 0.0)
                for q in poses[i+1:]:
                    self.assertGreaterEqual(max(abs(p[0]-q[0]),abs(p[1]-q[1])), BOX_SIZE_M+MIN_BOX_GAP_M-1e-9)

    def test_random_endpoints_have_clear_keepout_zones(self):
        for episode in self.episodes.values():
            self.assertTrue(point_is_clear(episode.start[:2],episode.layout))
            self.assertTrue(point_is_clear(episode.goal,episode.layout,GOAL_CLEARANCE_M))
            self.assertGreaterEqual(episode.start[2],-math.pi)
            self.assertLessEqual(episode.start[2],math.pi)

    def test_distance_ranges_prevent_trivial_success(self):
        for (level,_), episode in self.episodes.items():
            low, high=CURRICULUM_DISTANCE_LIMITS_M[level]
            distance=math.dist(episode.start[:2],episode.goal)
            self.assertGreaterEqual(distance,low)
            self.assertLessEqual(distance,high)
            self.assertFalse(goal_region_reached(episode.start[:2], episode.goal))

    def test_every_task_has_an_inflated_grid_path(self):
        for episode in self.episodes.values():
            self.assertTrue(navigation_path_exists(episode.start[:2],episode.goal,episode.layout,
                region_center=episode.region_center, region_size_m=episode.region_size_m))

    def test_blockage_metadata_matches_geometry(self):
        for episode in self.episodes.values():
            self.assertEqual(episode.straight_path_blocked,not straight_path_is_clear(episode.start[:2],episode.goal,episode.layout))
        for episode in self.episodes.values():
            self.assertEqual(episode.straight_path_blocker_count,
                             straight_path_blocker_count(episode.start[:2],episode.goal,episode.layout))
        self.assertTrue(any(e.straight_path_blocked for (level, _), e in self.episodes.items() if level == 2))
        self.assertTrue(any(not e.straight_path_blocked for (level, _), e in self.episodes.items() if level == 0))

    def test_segment_blocked_but_a_detour_exists(self):
        layout={BOX_NAMES[0]:(0,0,.4,0)}
        self.assertFalse(straight_path_is_clear((-2,0),(2,0),layout))
        self.assertTrue(navigation_path_exists((-2,0),(2,0),layout))
        self.assertTrue(straight_path_is_clear((-2,2),(2,2),layout))

    def test_impossible_barrier_rejected(self):
        layout={BOX_NAMES[i]:(0,float(y),.4,0) for i,y in enumerate(range(-4,5))}
        self.assertFalse(navigation_path_exists((-2,0),(2,0),layout))

    def test_invalid_geometry_rejected(self):
        for resolution in (0,-1,float('nan')):
            with self.assertRaises(ValueError): navigation_path_exists((0,0),(1,1),{},resolution)
        with self.assertRaises(ValueError): navigation_path_exists((0,0),(1,1),{'unknown':(0,0,.4,0)})
        with self.assertRaises(ValueError): navigation_path_exists((0,0),(1,1),{BOX_NAMES[0]:(0,0,.4,.2)})
        self.assertFalse(navigation_path_exists((8,0),(1,1),{}))

    def test_small_circular_goal_matches_visible_marker(self):
        self.assertTrue(goal_region_reached((.24,0),(0,0)))
        self.assertTrue(goal_region_reached((0,GOAL_SUCCESS_RADIUS_M),(0,0)))
        self.assertFalse(goal_region_reached((GOAL_SUCCESS_RADIUS_M+.001,0),(0,0)))
        self.assertFalse(goal_region_reached((.2,.2),(0,0)))

    def test_levels_change_task_geometry(self):
        self.assertNotEqual(sample_episode(10000,0),sample_episode(10000,2))
        self.assertEqual([sample_episode(10000,l).region_size_m for l in (0,1,2)], [3,6,9])

    def test_old_checkpoint_rejected(self):
        ensure_checkpoint_environment({'environment_id':ENVIRONMENT_ID})
        for metadata in ({},{'environment_id':'office'},{'environment_id':'square_arena_9m_diagonal_fixed_15boxes_circle_goal_v2'}):
            with self.assertRaises(ValueError): ensure_checkpoint_environment(metadata)

    def test_sdf_matches_runtime_geometry(self):
        path = Path(__file__).resolve().parents[2] / 'description_new/worlds/rl_square_arena.sdf'
        world = ET.parse(path).getroot().find('world')
        self.assertEqual(world.attrib['name'], WORLD_NAME)
        models = {model.attrib['name']: model for model in world.findall('model')}
        self.assertEqual({name for name in models if name.startswith('arena_box_')}, set(BOX_NAMES))
        episode = self.episodes[0, 10000]
        for index, name in enumerate(BOX_NAMES):
            size = models[name].findtext('link/collision/geometry/box/size')
            self.assertEqual(tuple(map(float, size.split())), (1, 1, .8))
            pose = tuple(map(float, models[name].findtext('pose').split()))
            self.assertEqual((pose[0], pose[1], pose[2], pose[5]),
                             episode.layout.get(name, parked_box_pose(index)))
        self.assertIsNone(models['goal_marker'].find('link/collision'))
        self.assertAlmostEqual(float(models['goal_marker'].findtext('link/visual/geometry/cylinder/radius')), GOAL_RADIUS_M)
        self.assertEqual(tuple(map(float, models['goal_marker'].findtext('pose').split()))[:2], episode.goal)


    def test_observation_is_lidar_and_final_goal_only(self):
        lidar,nearest=preprocess_scan([float('inf')]*360)
        distance,bearing=relative_goal(0,0,0,3,4)
        observation=build_observation(lidar,distance,bearing,0,0)
        self.assertEqual(observation.shape,(OBSERVATION_SIZE,))
        self.assertEqual(observation.dtype,np.float32)
        self.assertAlmostEqual(float(observation[72]),5/13,places=6)
        self.assertAlmostEqual(float(observation[73]),.8,places=6)
        self.assertAlmostEqual(float(observation[74]),.6,places=6)
        self.assertEqual(nearest,8.0)

    def test_scan_preserves_nearest_obstacle(self):
        scan=np.full(360,8.0);scan[7]=.3
        _,nearest=preprocess_scan(scan)
        self.assertAlmostEqual(nearest,.3,places=6)
        scan[:37]=float('nan')
        with self.assertRaises(InvalidScanError):preprocess_scan(scan)

    def test_actions_are_only_scaled_and_bounded(self):
        for action in ((-1,0),(0,.5),(1,-1),(-.4,.7)):
            v,w=policy_action_to_velocity(action)
            np.testing.assert_allclose(velocity_to_policy_action(v,w),action,atol=1e-6)
        self.assertAlmostEqual(policy_action_to_velocity((1,0))[0],MAX_LINEAR_SPEED_MPS,places=6)

    def test_reward_has_no_waypoint_bonus_or_heading_incentive(self):
        reward,parts=calculate_reward(2,1.98,0,1,.1,'running')
        self.assertNotIn('waypoint_reward',parts)
        self.assertEqual(parts['heading_reward'],0)
        self.assertAlmostEqual(reward,sum(parts[k] for k in ('progress_reward','heading_reward','step_penalty','obstacle_penalty','stuck_penalty','terminal_reward')))
        self.assertGreater(reward,0)

    def test_collision_priority_circular_success_and_timeout(self):
        self.assertEqual(determine_episode_outcome(.24,1,1,100,goal_reached=True),(True,False,'success'))
        self.assertEqual(determine_episode_outcome(.05,COLLISION_DISTANCE_M,1,100,goal_reached=True),(True,False,'collision'))
        self.assertEqual(determine_episode_outcome(2,1,100,100,goal_reached=False),(False,True,'timeout'))





"""Mocked ROS contract checks. No ROS transport or Gazebo physics is exercised."""
from collections import deque
import importlib
import math
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

from nav_learning.nav_math import policy_action_to_velocity, quaternion_from_yaw


def load_environment_with_ros_stubs():
    modules={}
    for name in ('rclpy','rclpy.executors','rclpy.node','rclpy.qos',
                 'geometry_msgs','geometry_msgs.msg','nav_msgs','nav_msgs.msg',
                 'ros_gz_interfaces','ros_gz_interfaces.msg','ros_gz_interfaces.srv',
                 'sensor_msgs','sensor_msgs.msg','rosgraph_msgs','rosgraph_msgs.msg'):
        modules[name]=ModuleType(name)
    for module,name in (('rclpy.node','Node'),('rclpy.executors','SingleThreadedExecutor'),
                        ('geometry_msgs.msg','Twist'),('nav_msgs.msg','Odometry'),
                        ('ros_gz_interfaces.msg','Entity'),('ros_gz_interfaces.srv','SetEntityPose'),
                        ('ros_gz_interfaces.srv','ControlWorld'),('rosgraph_msgs.msg','Clock'),
                        ('sensor_msgs.msg','LaserScan')):
        setattr(modules[module],name,type(name,(),{}))
    modules['rclpy.qos'].qos_profile_sensor_data=object()
    with patch.dict(sys.modules,modules):
        # Force this test copy to bind stubs even if another test imported real ROS.
        previous=sys.modules.pop('nav_learning.ros_nav_env',None)
        try:module=importlib.import_module('nav_learning.ros_nav_env')
        finally:
            sys.modules.pop('nav_learning.ros_nav_env',None)
            if previous is not None:sys.modules['nav_learning.ros_nav_env']=previous
    return module


ENV=load_environment_with_ros_stubs()


def messages(pose=(0,0,0), nearest=8.0):
    x,y,yaw=pose
    qx,qy,qz,qw=quaternion_from_yaw(yaw)
    stamp=NS(sec=1,nanosec=0)
    scan=NS(header=NS(stamp=stamp),range_min=.15,ranges=[nearest]*360)
    odom=NS(header=NS(stamp=stamp),pose=NS(pose=NS(position=NS(x=x,y=y),
            orientation=NS(x=qx,y=qy,z=qz,w=qw))),
            twist=NS(twist=NS(linear=NS(x=0),angular=NS(z=0))))
    return scan,odom


def bare_environment():
    e=object.__new__(ENV.RosNavEnv)
    e._reward_settings = ENV.RewardSettings()
    e._action_settings = ENV.ActionSettings()
    e._detour_distance = None
    e._previous_detour_distance_m = None
    e._sensor_condition=threading.Condition()
    e._position_history=deque(maxlen=80)
    e._episode_active=True;e._curriculum_level=0;e._previous_goal_distance_m=2
    e._scan_count=e._odom_count=1
    e._scan_stamp_ns=e._odom_stamp_ns=1_000_000_000
    e._clock_stamp_ns=e._paused_stamp_ns=1_000_000_000
    e._last_step_timing={}
    e._step_count=0;e.goal_xy=(3,4)
    e.stops=0
    def stop():e.stops+=1
    e.publish_stop=stop
    e.pause_simulation=lambda:None
    e.get_logger=lambda:NS(warning=lambda text:None)
    e.teleports=[]
    def teleport(entity_name,x,y,z,yaw):
        e.teleports.append((entity_name,x,y,z,yaw))
        if entity_name=='hv_bot':e.robot_pose=(x,y,yaw)
    e._set_entity_pose=teleport
    e._wait_for_fresh_sensors=lambda **kwargs:messages(e.robot_pose)
    e._wait_for_sensor_time=lambda *args,**kwargs:messages(e.robot_pose)
    def advance(duration):
        e._last_step_timing=dict(simulation_interval_sec=duration,maximum_sensor_age_sec=0)
        return e._wait_for_sensor_time(target_stamp_ns=1,timeout_sec=1)
    e._advance_simulation=advance
    return e


class RosContractTests(unittest.TestCase):
    def test_new_reward_and_action_profiles_reach_the_ros_step(self):
        e = bare_environment()
        e._reward_settings = ENV.RewardSettings(progress_mode="detour_potential")
        e._action_settings = ENV.ActionSettings(stop_fraction=0.15)
        _, reset_info = e.reset(70000, 1, 2, corner_probability=1.0, mirror_pair=True)
        self.assertEqual(reset_info['task_kind'], 'corner')
        self.assertIsNotNone(e._previous_detour_distance_m)
        commands = []
        e._publish_velocity_command = lambda **kwargs: commands.append(kwargs)
        _, reward, terminated, truncated, info = e.step([-0.85, 0.5])
        self.assertEqual(commands[-1]['linear_velocity_mps'], 0)
        self.assertAlmostEqual(commands[-1]['angular_velocity_radps'], 0.6)
        self.assertTrue(math.isfinite(reward))
        self.assertIsNotNone(info['detour_distance_m'])
        self.assertFalse(terminated or truncated)

    def test_sensor_observation_targets_final_goal(self):
        e=bare_environment()
        observation,info=e._observation_from_messages(*messages())
        self.assertAlmostEqual(float(observation[72]),5/13,places=6)
        self.assertAlmostEqual(float(observation[73]),.8,places=6)
        self.assertAlmostEqual(float(observation[74]),.6,places=6)
        self.assertNotIn('route_waypoint_count',info)
        self.assertNotIn('distance_to_target_m',info)

    def test_ros_success_region_is_circular(self):
        e=bare_environment();e.goal_xy=(0,0)
        _,inside=e._observation_from_messages(*messages((.24,0,0)))
        _,outside=e._observation_from_messages(*messages((.2,.2,0)))
        self.assertTrue(inside['goal_region_reached'])
        self.assertFalse(outside['goal_region_reached'])

    def test_reset_stages_robot_moves_every_box_and_robot_last(self):
        e=bare_environment()
        with patch.object(ENV.time,'sleep'):
            observation,info=e.reset(10000,2)
        self.assertEqual(e.teleports[0][0],'hv_bot')
        self.assertEqual(e.teleports[0][2],-6.0)
        self.assertEqual([p[0] for p in e.teleports[1:16]],list(ENV.BOX_NAMES))
        self.assertEqual(e.teleports[-2][0],'goal_marker')
        self.assertEqual(e.teleports[-1][0],'hv_bot')
        self.assertEqual(info['randomized_object_count'],len(ENV.sample_episode(10000,2).layout))
        self.assertEqual(observation.shape,(77,))
        self.assertTrue(e._episode_active)
        self.assertEqual(e.goal_xy,info['goal'])
        self.assertGreaterEqual(info['distance_to_goal_m'],3)
        self.assertGreaterEqual(info['straight_path_blocker_count'],1)
        self.assertEqual(e._previous_goal_distance_m,info['distance_to_goal_m'])

    def test_sparse_reset_parks_all_unused_boxes_after_dense_reset(self):
        e=bare_environment()
        with patch.object(ENV.time,'sleep'):
            e.reset(10000,2)
            e.teleports=[]
            _,info=e.reset(10000,0)
        task=ENV.sample_episode(10000,0)
        boxes={p[0]:p[1:] for p in e.teleports if p[0] in ENV.BOX_NAMES}
        self.assertEqual(set(boxes),set(ENV.BOX_NAMES))
        self.assertEqual(info['randomized_object_count'],len(task.layout))
        self.assertEqual(set(info['object_layout']),set(task.layout))
        for index,name in enumerate(ENV.BOX_NAMES):
            self.assertEqual(boxes[name], task.layout.get(name,ENV.parked_box_pose(index)))

    def test_reset_failure_invalidates_previous_episode(self):
        e=bare_environment()
        e._set_entity_pose=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError('service failed'))
        with patch.object(ENV.time,'sleep'):
            with self.assertRaises(RuntimeError):e.reset(10000,2)
        self.assertFalse(e._episode_active)
        self.assertIsNone(e._curriculum_level)
        self.assertIsNone(e.goal_xy)
        with self.assertRaises(RuntimeError):e.step([1,0])

    def test_no_action_override_in_dense_scan(self):
        e=bare_environment();e.robot_pose=(0,0,0)
        command=[]
        e._publish_velocity_command=lambda **kwargs:command.append(kwargs)
        e._wait_for_sensor_time=lambda **kwargs:messages(nearest=.24)
        action=np.asarray([.4,-.7],dtype=np.float32)
        _,info=e._apply_action_and_observe(action)
        v,w=policy_action_to_velocity(action)
        self.assertEqual(command,[dict(linear_velocity_mps=v,angular_velocity_radps=w)])
        np.testing.assert_allclose(info['executed_action'],action,atol=1e-6)
        self.assertEqual(info['step_count'],1)

    def test_failed_sensor_wait_stops_robot(self):
        e=bare_environment()
        e._publish_velocity_command=lambda **kwargs:None
        e._wait_for_sensor_time=lambda **kwargs: (_ for _ in ()).throw(TimeoutError('sensor'))
        with self.assertRaises(TimeoutError):e._apply_action_and_observe([1,0])
        self.assertFalse(e._episode_active)
        self.assertGreater(e.stops,0)

    def test_step_uses_circular_success_test_and_final_goal_reward(self):
        e=bare_environment();e._previous_goal_distance_m=.50
        def transition(action):
            e._step_count+=1
            return np.ones(77,dtype=np.float32),dict(distance_to_goal_m=.24,
                nearest_obstacle_m=1,goal_region_reached=True,relative_goal_bearing_rad=0,
                commanded_linear_velocity_mps=.1,stuck=False)
        e._apply_action_and_observe=transition
        _,reward,terminated,truncated,info=e.step([0,0])
        self.assertTrue(terminated);self.assertFalse(truncated)
        self.assertEqual(info['outcome'],'success')
        self.assertAlmostEqual(info['reward_components']['progress_m'],.50-.24)
        self.assertGreater(reward,25)
        self.assertFalse(e._episode_active)
        self.assertGreater(e.stops,0)




"""Evaluator wiring checks using fake sensor transitions and a fake policy."""
from contextlib import nullcontext
import csv
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from nav_learning.nav_math import ENVIRONMENT_ID, sample_episode
from nav_learning.nav_math import build_observation


class FakeEnvironment:
    def __init__(self, reward_settings=None, action_settings=None):
        self.reward_settings = reward_settings
        self.action_settings = action_settings
        type(self).last_reward_settings = reward_settings
        self.simulation_time_ns=0
    def assert_simulation_time_unchanged(self,expected):
        if self.simulation_time_ns != expected:raise RuntimeError('Simulation drift')
    def pause_simulation(self):pass
    def reset(self,seed,curriculum_level,practice_phase=None,**options):
        self.task=sample_episode(seed,curriculum_level,practice_phase,**options); self.steps=0
        self.robot_x,self.robot_y=self.task.start[:2]
        gx,gy=self.task.goal
        self.initial=float(np.hypot(gx-self.robot_x,gy-self.robot_y))
        obs=build_observation(np.ones(72,dtype=np.float32),self.initial,0,0,0)
        info=dict(goal=self.task.goal,requested_start=self.task.start,object_layout=self.task.layout,
                  task_kind=self.task.task_kind,mirrored=self.task.mirrored,
                  straight_path_blocked=self.task.straight_path_blocked,randomized_object_count=len(self.task.layout),
                  straight_path_blocker_count=self.task.straight_path_blocker_count,
                  robot_x=self.robot_x,robot_y=self.robot_y,distance_to_goal_m=self.initial)
        info.update(region_center=self.task.region_center,region_size_m=self.task.region_size_m)
        return obs,info
    def step(self,action):
        self.steps+=1
        self.simulation_time_ns+=100_000_000
        self.robot_x+=.01
        info=dict(robot_x=self.robot_x,robot_y=self.robot_y,step_count=self.steps,
                  distance_to_goal_m=self.initial-.005*self.steps,nearest_obstacle_m=.5,
                  commanded_linear_velocity_mps=.1,commanded_angular_velocity_radps=0,
                  stuck=False,outcome='success' if self.steps==2 else 'running',
                  simulation_interval_sec=.1,maximum_sensor_age_sec=.03)
        return np.ones(77,dtype=np.float32),.1,self.steps==2,False,info
    def close(self):pass


class FakeTensor:
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return np.asarray([0,0],dtype=np.float32)


def evaluator_with_stubs(metadata=None):
    torch=ModuleType('torch'); torch.manual_seed=lambda seed:None; torch.inference_mode=nullcontext
    ppo=ModuleType('nav_learning.ppo'); ppo.choose_device=lambda x:x
    ppo.observation_to_tensor=lambda x,device:x
    model=NS(eval=lambda:None,deterministic_action=lambda x:FakeTensor())
    ppo.load_checkpoint=lambda *args,**kwargs:(NS(model=model),dict(metadata=metadata if metadata is not None else {'environment_id':ENVIRONMENT_ID},training_state={}))
    ros=ModuleType('rclpy');ros.init=lambda **kwargs:None;ros.ok=lambda:True;ros.shutdown=lambda:None
    env=ModuleType('nav_learning.ros_nav_env');env.RosNavEnv=FakeEnvironment;env.MAX_EPISODE_STEPS={0:450,1:1000,2:1600}
    modules={'torch':torch,'nav_learning.ppo':ppo,'rclpy':ros,'nav_learning.ros_nav_env':env}
    with patch.dict(sys.modules,modules):
        previous=sys.modules.pop('nav_learning.evaluate_ppo',None)
        try:module=importlib.import_module('nav_learning.evaluate_ppo')
        finally:
            sys.modules.pop('nav_learning.evaluate_ppo',None)
            if previous is not None:sys.modules['nav_learning.evaluate_ppo']=previous
    return module,modules


class EvaluationContractTests(unittest.TestCase):
    def test_corner_evaluation_uses_pairs_and_checkpoint_action_mapping(self):
        module, modules = evaluator_with_stubs(metadata={
            "environment_id": ENVIRONMENT_ID, "action_settings": {"stop_fraction": 0.15}})
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            root = Path(directory)
            summary = module.run_evaluation(Path('new.pt'), root, (1,), 2, 70000,
                                           'cpu', False, 2, practice_phase=2, task_suite='corners')
            layouts = json.loads((root/'evaluation_layouts.json').read_text())['episodes']
            self.assertEqual(summary['action_settings']['stop_fraction'], 0.15)
            self.assertEqual(summary['task_distribution'], 'mirrored_corners')
            self.assertEqual(layouts[0]['goal'][0], layouts[1]['goal'][0])
            self.assertEqual(layouts[0]['goal'][1], -layouts[1]['goal'][1])

    def test_practice_evaluation_is_explicit_and_records_the_right_distribution(self):
        module,modules=evaluator_with_stubs()
        args=module._parse_arguments(['--checkpoint','x.pt','--levels','1','--practice-phase','0'])
        self.assertEqual(args.practice_phase,0)
        with self.assertRaises(SystemExit):
            module._parse_arguments(['--checkpoint','x.pt','--practice-phase','0'])
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules):
            root=Path(directory)
            summary=module.run_evaluation(Path('fake.pt'),root,(1,),2,50000,'cpu',False,2,
                                          practice_phase=0)
            self.assertEqual(summary['task_distribution'],'detour_practice_0')
            with (root/'evaluation_episodes.csv').open() as stream:rows=list(csv.DictReader(stream))
            self.assertTrue(all(int(row['randomized_object_count'])==1 for row in rows))
            self.assertTrue(all(int(row['practice_phase'])==0 for row in rows))
            layouts=json.loads((root/'evaluation_layouts.json').read_text())
            self.assertTrue(all(row['practice_phase']==0 for row in layouts['episodes']))

    def test_evaluator_uses_checkpoint_reward_profile(self):
        reward = {"safety_distance_m": 0.5, "near_obstacle_penalty_scale": 0.2}
        module, modules = evaluator_with_stubs(metadata={
            "environment_id": ENVIRONMENT_ID, "reward_settings": reward})
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            summary = module.run_evaluation(Path('tuned.pt'), Path(directory),
                                            (1,), 1, 50000, 'cpu', False, 1)
        self.assertEqual(summary['reward_settings']['safety_distance_m'], reward['safety_distance_m'])
        self.assertEqual(summary['reward_settings']['near_obstacle_penalty_scale'], reward['near_obstacle_penalty_scale'])
        self.assertEqual(summary['reward_settings']['progress_mode'], 'euclidean')
        self.assertEqual(FakeEnvironment.last_reward_settings.safety_distance_m, 0.5)

    def test_baseline_ignores_lidar(self):
        module, _ = evaluator_with_stubs()
        observation = build_observation(np.ones(72, dtype=np.float32), 2, 0, 0, 0)
        action = module.goal_seeking_action(observation)
        observation[:72] = 0
        np.testing.assert_array_equal(module.goal_seeking_action(observation), action)
        np.testing.assert_array_equal(action, [1, 0])

    def test_baseline_and_policy_use_identical_tasks_and_save_layouts(self):
        module,modules=evaluator_with_stubs()
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules):
            root=Path(directory)
            for baseline in (False,True):
                path=root/str(baseline)
                summary=module.run_evaluation(None if baseline else Path('fake.pt'),path,(2,),2,70000,'cpu',False,2,baseline)
                self.assertEqual(summary['overall']['success_rate'],1)
                self.assertEqual(summary['environment_id'],ENVIRONMENT_ID)
                self.assertLessEqual(summary['overall']['mean_success_path_efficiency'],1)
                with (path/'evaluation_episodes.csv').open() as stream:
                    rows=list(csv.DictReader(stream))
                self.assertEqual([int(row['seed']) for row in rows],[90000,90001])
                self.assertTrue(all(12 <= int(row['randomized_object_count']) <= 15 for row in rows))
                self.assertTrue(all(float(row['region_size_m'])==9 for row in rows))
                self.assertNotIn('route_waypoint_count',rows[0])
            a=json.loads((root/'False/evaluation_layouts.json').read_text())
            b=json.loads((root/'True/evaluation_layouts.json').read_text())
            self.assertEqual(a,b)

    def test_evaluator_rejects_office_checkpoint(self):
        module,modules=evaluator_with_stubs(metadata={})
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules):
            with self.assertRaises(ValueError):
                module.run_evaluation(Path('old.pt'),Path(directory),(2,),1,70000,'cpu',False,1)

    def test_cli_requires_either_baseline_or_checkpoint(self):
        module,_=evaluator_with_stubs()
        self.assertEqual(module._parse_arguments(['--baseline','goal-seeking']).baseline,'goal-seeking')
        with self.assertRaises(SystemExit):module._parse_arguments([])
        with self.assertRaises(SystemExit):module._parse_arguments(['--checkpoint','x.pt','--baseline','goal-seeking'])




"""Check training statistics without importing the real PyTorch optimizer."""
import importlib
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
from contextlib import nullcontext, redirect_stdout
import csv
import io
from pathlib import Path
import tempfile

import numpy as np


def trainer_with_stubs():
    torch=ModuleType('torch')
    ppo=ModuleType('nav_learning.ppo')
    for name in ('ActorCritic','PPOAgent','RolloutBuffer','PPOHyperparameters'):
        setattr(ppo,name,type(name,(),{}))
    for name in ('choose_device','load_checkpoint','observation_to_tensor','save_checkpoint'):
        setattr(ppo,name,lambda *args,**kwargs:None)
    modules={'torch':torch,'nav_learning.ppo':ppo}
    with patch.dict(sys.modules,modules):
        previous=sys.modules.pop('nav_learning.train_ppo',None)
        try:module=importlib.import_module('nav_learning.train_ppo')
        finally:
            sys.modules.pop('nav_learning.train_ppo',None)
            if previous is not None:sys.modules['nav_learning.train_ppo']=previous
    return module


TRAIN=trainer_with_stubs()


class TrainingContractTests(unittest.TestCase):
    def test_appending_new_diagnostics_keeps_legacy_csv_columns_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'updates.csv'
            path.write_text('update,entropy\n1,3.0\n')
            logger = TRAIN.CsvLogger(path, ('update', 'entropy', 'squashed_entropy'))
            logger.write(dict(update=2, entropy=2.0, squashed_entropy=0.5))
            logger.close()
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [{'update': '1', 'entropy': '3.0'},
                                    {'update': '2', 'entropy': '2.0'}])

    def setUp(self):
        self.state={'recent_successes':[],'curriculum_level':2}
        self.settings=NS(curriculum=NS(enabled=False,promotion_window=3,minimum_episodes=3,max_level=2,success_threshold=.7))

    def test_disabled_curriculum_still_logs_rolling_success(self):
        for success in (True,False,True):
            rate,promoted=TRAIN._update_curriculum_after_episode(self.state,self.settings,success,None)
            self.assertFalse(promoted)
        self.assertAlmostEqual(rate,2/3)
        self.assertEqual(self.state['recent_successes'],[1,0,1])
        self.assertEqual(self.state['curriculum_level'],2)

    def test_fixed_level_keeps_success_statistics(self):
        for success in (True,False):
            rate,promoted=TRAIN._update_curriculum_after_episode(self.state,self.settings,success,2)
        self.assertEqual(rate,.5)
        self.assertFalse(promoted)
        self.assertEqual(self.state['recent_successes'],[1,0])

    def test_recent_window_is_bounded(self):
        for success in (True,True,True,False):
            rate,_=TRAIN._update_curriculum_after_episode(self.state,self.settings,success,None)
        self.assertEqual(self.state['recent_successes'],[1,1,0])
        self.assertAlmostEqual(rate,2/3)

    def test_rehearsal_cannot_inflate_promotion_success(self):
        self.state['recent_successes']=[0,0,0]
        rate,ready=TRAIN._update_curriculum_after_episode(self.state,self.settings,True,None,False)
        self.assertEqual(rate,0)
        self.assertFalse(ready)
        self.assertEqual(self.state['recent_successes'],[0,0,0])

    def test_full_window_triggers_check_before_promotion(self):
        self.state['curriculum_level']=0
        self.settings.curriculum.enabled=True
        for _ in range(2):
            _,ready=TRAIN._update_curriculum_after_episode(self.state,self.settings,True,None)
            self.assertFalse(ready)
        _,ready=TRAIN._update_curriculum_after_episode(self.state,self.settings,True,None)
        self.assertTrue(ready)
        self.assertEqual(self.state['curriculum_level'],0)

    def test_failed_validation_waits_for_more_current_stage_episodes(self):
        self.state['curriculum_level']=0
        self.settings.curriculum.enabled=True
        self.settings.curriculum.validation_success_threshold=.8
        self.settings.curriculum.validation_retry_episodes=3
        self.state['recent_successes']=[1,1,1]
        self.assertFalse(TRAIN._complete_promotion_validation(self.state,self.settings,.7))
        for index in range(3):
            _,ready=TRAIN._update_curriculum_after_episode(self.state,self.settings,True,None)
            self.assertEqual(ready,index==2)

    def test_passed_validation_promotes_and_clears_old_successes(self):
        self.state['curriculum_level']=0
        self.settings.curriculum.validation_success_threshold=.8
        self.assertTrue(TRAIN._complete_promotion_validation(self.state,self.settings,.8))
        self.assertEqual(self.state['curriculum_level'],1)
        self.assertEqual(self.state['recent_successes'],[])

    def test_existing_fresh_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'metrics').mkdir()
            (root/'metrics/train_episodes.csv').write_text('episode\n1\n')
            with self.assertRaises(FileExistsError):
                TRAIN._resolve_run_directory(self.settings,None,str(root))

    def test_training_loop_promotes_validates_and_keeps_step_counts_separate(self):
        self._exercise_training_loop()

    def test_training_loop_finishes_all_detour_phases_before_stage3(self):
        self._exercise_training_loop(detour_practice=True)

    def test_training_loop_runs_mirrored_corner_tasks_with_new_controls(self):
        self._exercise_training_loop(detour_practice=True, corner_profile=True)

    def test_clock_drift_during_optimizer_update_stops_training_and_saves(self):
        self._exercise_training_loop(corrupt_clock=True)

    def _exercise_training_loop(self,detour_practice=False,corrupt_clock=False,corner_profile=False):
        """Exercise real trainer control flow with mocked ROS and tensor math."""
        class Environment(FakeEnvironment):
            latest=None
            audits=0
            def __init__(self,**kwargs):
                super().__init__(**kwargs)
                Environment.latest=self
            def assert_simulation_time_unchanged(self,expected):
                Environment.audits+=1
                super().assert_simulation_time_unchanged(expected)
            def publish_stop(self): pass
        class Model:
            def to(self,device): return self
            def eval(self): pass
            def sample_action(self,observation): return FakeTensor(),0,0
            def deterministic_action(self,observation): return FakeTensor()
            def predict_value(self,observation): return 0
        class Buffer:
            def __init__(self,*args,**kwargs): pass
            def add(self,**kwargs): pass
            def compute_returns_and_advantages(self,**kwargs): return {}
        hp=NS(discount_factor=.99,gae_lambda=.95,learning_rate=.0003,entropy_coefficient=.001)
        def update(rollout):
            if corrupt_clock:Environment.latest.simulation_time_ns+=100_000_000
            return dict(policy_loss=0,value_loss=0,approximate_kl=0)
        agent=NS(model=Model(),hyperparameters=hp,update=update)
        budget=100 if detour_practice else 40
        settings=NS(device='cpu',seed=7,reset_seed_base=10000,total_timesteps=budget,
                    reward=TRAIN.RewardSettings(), action=TRAIN.ActionSettings(),
                    rollout_steps=10,ppo=hp,anneal_learning_rate=False,
                    checkpoint_every_updates=2,log_every_updates=1,
                    curriculum=TRAIN.CurriculumSettings(start_level=0,max_level=2,enabled=True,
                        detour_practice=detour_practice,detour_success_threshold=.9,
                        corner_practice_probability=0.0,mirror_training=False,
                        promotion_window=3,minimum_episodes=3,success_threshold=.8,
                        validation_episodes=2,validation_success_threshold=.8,
                        validation_retry_episodes=3))
        if corner_profile:
            settings.action = TRAIN.ActionSettings(stop_fraction=0.15)
            settings.reward = TRAIN.RewardSettings(progress_mode='detour_potential')
            from dataclasses import replace
            settings.curriculum = replace(settings.curriculum,
                corner_practice_probability=1.0, mirror_training=True)
        ros=ModuleType('rclpy')
        ros.init=lambda **kwargs:None; ros.ok=lambda:True; ros.shutdown=lambda:None
        env=ModuleType('nav_learning.ros_nav_env'); env.RosNavEnv=Environment
        checkpoints={}
        def save(root,name,agent,state,config):
            checkpoints[name]=dict(state)
            return root/'checkpoints'/name
        torch=NS(no_grad=nullcontext,inference_mode=nullcontext)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root=Path(directory)
            with patch.dict(sys.modules,{'rclpy':ros,'nav_learning.ros_nav_env':env}), \
                 patch.multiple(TRAIN,torch=torch,ActorCritic=Model,
                    PPOAgent=lambda *args:agent,RolloutBuffer=Buffer,
                    _set_global_seeds=lambda seed:None,
                    _save_training_checkpoint=save):
                if corrupt_clock:
                    with self.assertRaisesRegex(RuntimeError,'Simulation drift'):
                        TRAIN.run_training(settings,root/'config.yaml',root,None,None)
                    self.assertIn('crash_recovery.pt',checkpoints)
                    self.assertNotIn('final.pt',checkpoints)
                    return
                TRAIN.run_training(settings,root/'config.yaml',root,None,None)
            state=checkpoints['final.pt']
            self.assertEqual(state['curriculum_level'],2)
            self.assertEqual(state['total_timesteps'],budget)
            self.assertEqual(state['validation_steps'],20 if detour_practice else 8)
            self.assertGreaterEqual(Environment.audits,2*budget//10)
            if detour_practice:
                self.assertEqual(state['detour_phase'],3)
                for phase in (1,2,3):self.assertIn(f'detour_phase_{phase}_complete.pt',checkpoints)
            self.assertIn('stage_1_complete.pt',checkpoints)
            self.assertIn('stage_2_complete.pt',checkpoints)
            with (root/'metrics/promotion_checks.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),10 if detour_practice else 4)
            self.assertTrue(all(int(row['seed'])>=1_000_000_000 for row in rows))
            with (root/'metrics/train_episodes.csv').open() as stream:
                episodes=list(csv.DictReader(stream))
            self.assertEqual(len(episodes),budget//2)
            if corner_profile:
                self.assertEqual(state['action_settings']['stop_fraction'], 0.15)
                stage2 = [row for row in episodes if row['curriculum_level'] == '1']
                self.assertTrue(stage2)
                self.assertTrue(all(row['task_kind'] == 'corner' for row in stage2))
                self.assertEqual({row['mirrored'] for row in stage2}, {'0', '1'})
            self.assertTrue(any(int(row['rehearsal']) for row in episodes))




"""Check that easier episodes and promotions are represented honestly."""
import unittest
import numpy as np

from nav_learning.plot_ppo_results import _current_stage_success


class PlotContractTests(unittest.TestCase):
    def test_success_curve_excludes_rehearsal_and_resets_at_promotion(self):
        rows=[dict(training_stage='0',success='1',rehearsal='0'),
              dict(training_stage='0',success='0',rehearsal='0'),
              dict(training_stage='1',success='1',rehearsal='1'),
              dict(training_stage='1',success='0',rehearsal='0'),
              dict(training_stage='1',success='1',rehearsal='1'),
              dict(training_stage='1',success='1',rehearsal='0')]
        result=_current_stage_success(rows)
        np.testing.assert_allclose(result[[0,1,3,4,5]], [1,.5,0,0,.5])
        self.assertTrue(np.isnan(result[2]))




class SimulatorTimingTests(unittest.TestCase):
    def test_unresponsive_control_service_times_out_and_cancels_request(self):
        from concurrent.futures import Future
        e=bare_environment()
        future=Future()
        e.control_world_client=NS(call_async=lambda _:future)
        def request():return NS(world_control=NS(pause=True,run_to_sim_time=NS(sec=0,nanosec=0)))
        with patch.object(ENV.ControlWorld,'Request',request,create=True), \
             patch.object(ENV,'STEP_SENSOR_TIMEOUT_SEC',.01):
            with self.assertRaises(TimeoutError):ENV.RosNavEnv._control_world(e,pause=True)
        self.assertTrue(future.cancelled())

    def native_environment(self, overshoot_ns=0, stale=False):
        e = bare_environment()
        del e._advance_simulation
        del e._wait_for_sensor_time
        e._latest_scan, e._latest_odom = messages()
        e.control_requests = []
        def control(*, pause, run_to_ns=None):
            e.control_requests.append((pause, run_to_ns))
            def deliver():
                end = run_to_ns + overshoot_ns
                stamp = 1_000_000_000 if stale else run_to_ns - 20_000_000
                scan, odom = messages()
                for message in (scan, odom):
                    message.header.stamp = NS(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000)
                e._clock_callback(NS(clock=NS(sec=end // 1_000_000_000,nanosec=end % 1_000_000_000)))
                e._scan_callback(scan)
                e._odom_callback(odom)
            timer = threading.Timer(.005, deliver)
            timer.start()
        e._control_world = control
        return e

    def test_action_uses_native_run_to_time_and_returns_only_new_sensor_frames(self):
        e = self.native_environment()
        scan, odom = e._advance_simulation(.1)
        self.assertEqual(e.control_requests, [(False, 1_100_000_000)])
        self.assertEqual(e._paused_stamp_ns, 1_100_000_000)
        self.assertAlmostEqual(e._last_step_timing['simulation_interval_sec'], .1)
        self.assertAlmostEqual(e._last_step_timing['maximum_sensor_age_sec'], .02)
        e.assert_simulation_time_unchanged(1_100_000_000)

    def test_overshooting_clock_is_rejected(self):
        e = self.native_environment(overshoot_ns=20_000_000)
        with self.assertRaisesRegex(RuntimeError, 'overshot'):
            e._advance_simulation(.1)

    def test_stale_sensor_frames_cannot_complete_a_step(self):
        e = self.native_environment(stale=True)
        with patch.object(ENV, 'STEP_SENSOR_TIMEOUT_SEC', .04):
            with self.assertRaises(TimeoutError):e._advance_simulation(.1)

    def test_external_unpause_is_detected_before_the_next_step(self):
        e = self.native_environment()
        e._clock_stamp_ns += 100_000_000
        with self.assertRaisesRegex(RuntimeError, 'outside an environment step'):
            e._advance_simulation(.1)
        self.assertEqual(e.control_requests, [])

    def test_service_acceptance_is_checked_and_absolute_target_is_serialized(self):
        from concurrent.futures import Future
        e = bare_environment()
        requests = []
        def request():return NS(world_control=NS(pause=False,run_to_sim_time=NS(sec=0,nanosec=0)))
        def call(value):
            requests.append(value)
            result = Future();result.set_result(NS(success=True));return result
        e.control_world_client = NS(call_async=call)
        with patch.object(ENV.ControlWorld, 'Request', request, create=True):
            ENV.RosNavEnv._control_world(e,pause=False,run_to_ns=2_123_000_000)
        self.assertFalse(requests[0].world_control.pause)
        self.assertEqual(requests[0].world_control.run_to_sim_time.sec,2)
        self.assertEqual(requests[0].world_control.run_to_sim_time.nanosec,123_000_000)
        bad = Future();bad.set_result(NS(success=False))
        e.control_world_client = NS(call_async=lambda _:bad)
        with patch.object(ENV.ControlWorld, 'Request', request, create=True):
            with self.assertRaisesRegex(RuntimeError,'rejected'):
                ENV.RosNavEnv._control_world(e,pause=True)


class DetourPracticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {(phase,seed):sample_episode(seed,1,phase)
                     for phase in (0,1,2) for seed in range(60000,60012)}

    def test_practice_has_requested_box_count_and_a_blocked_but_reachable_goal(self):
        for (phase,_), task in self.tasks.items():
            self.assertEqual(len(task.layout), phase + 1)
            self.assertEqual(task.practice_phase, phase)
            self.assertEqual(task.region_size_m,6)
            self.assertGreaterEqual(task.straight_path_blocker_count,1)
            self.assertTrue(navigation_path_exists(task.start[:2],task.goal,task.layout,
                region_center=task.region_center,region_size_m=task.region_size_m))
            self.assertEqual(task,sample_episode(_,1,phase))

    def test_single_box_headings_focus_on_the_detour(self):
        for (phase,_),task in self.tasks.items():
            if phase == 0:
                heading=math.atan2(task.goal[1]-task.start[1],task.goal[0]-task.start[0])
                self.assertLessEqual(abs(math.atan2(math.sin(task.start[2]-heading),
                    math.cos(task.start[2]-heading))),math.pi/3+1e-9)

    def test_default_sampler_remains_the_full_benchmark(self):
        for seed in range(60000,60006):
            task=sample_episode(seed,1)
            self.assertEqual(task.practice_phase,-1)
            self.assertGreaterEqual(len(task.layout),5)
        with self.assertRaises(ValueError):sample_episode(1,0,0)
        with self.assertRaises(ValueError):sample_episode(1,1,3)

    def settings(self):
        return NS(curriculum=NS(enabled=True,promotion_window=100,minimum_episodes=100,
            corner_practice_probability=0.0,mirror_training=False,
            max_level=2,success_threshold=.8,validation_success_threshold=.8,
            validation_retry_episodes=25,detour_practice=True,detour_success_threshold=.9))

    def test_practice_requires_90_percent_and_separate_validation(self):
        state=dict(curriculum_level=1,detour_phase=0,recent_successes=[])
        settings=self.settings()
        for i in range(100):rate,ready=TRAIN._update_curriculum_after_episode(state,settings,i<89,None)
        self.assertFalse(ready)
        rate,ready=TRAIN._update_curriculum_after_episode(state,settings,True,None)
        self.assertLess(rate,.9)  # An old success leaves the rolling window.
        state['recent_successes']=[1]*89+[0]*10
        rate,ready=TRAIN._update_curriculum_after_episode(state,settings,True,None)
        self.assertAlmostEqual(rate,.9)
        self.assertTrue(ready)
        state['recent_successes']=[1]*90+[0]*10
        self.assertFalse(TRAIN._complete_promotion_validation(state,settings,.8))
        self.assertEqual(state['detour_phase'],0)
        self.assertTrue(TRAIN._complete_promotion_validation(state,settings,.9))
        self.assertEqual(state['detour_phase'],1)
        self.assertEqual(state['curriculum_level'],1)
        self.assertEqual(state['recent_successes'],[])

    def test_practice_progression_reaches_full_stage2_before_stage3(self):
        settings=self.settings()
        state=dict(curriculum_level=1,detour_phase=0,recent_successes=[])
        for expected in (1,2,3):
            self.assertTrue(TRAIN._complete_promotion_validation(state,settings,.9))
            self.assertEqual(state['detour_phase'],expected)
            self.assertEqual(state['curriculum_level'],1)
        self.assertTrue(TRAIN._complete_promotion_validation(state,settings,.8))
        self.assertEqual(state['curriculum_level'],2)

    def test_rehearsal_is_excluded_and_training_passes_practice_to_reset(self):
        settings=self.settings()
        seed=next(seed for seed in range(60000,60020) if np.random.default_rng(seed).random()>=.2)
        state=dict(curriculum_level=1,detour_phase=0,recent_successes=[],next_reset_seed=seed)
        env=FakeEnvironment()
        _,_,episode=TRAIN._reset_episode(env,state,None,settings)
        self.assertEqual(episode['practice_phase'],0)
        self.assertEqual(len(env.task.layout),1)
        TRAIN._update_curriculum_after_episode(state,settings,True,None,False)
        self.assertEqual(state['recent_successes'],[])

    def test_plot_success_resets_at_practice_changes(self):
        from nav_learning.plot_ppo_results import _current_stage_success
        rows=[dict(training_stage='1',training_practice_phase='0',success='1',rehearsal='0'),
              dict(training_stage='1',training_practice_phase='1',success='0',rehearsal='0')]
        np.testing.assert_allclose(_current_stage_success(rows),[1,0])


if __name__ == '__main__':
    unittest.main()
