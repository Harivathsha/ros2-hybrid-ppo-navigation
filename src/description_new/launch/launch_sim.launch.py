"""Launch the square arena, original robot and required ROS/Gazebo bridges."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('description_new')
    gz_share = get_package_share_directory('ros_gz_sim')
    world = LaunchConfiguration('world')
    world_name = LaunchConfiguration('world_name')
    use_sim_time = LaunchConfiguration('use_sim_time')
    resource_path = os.pathsep.join(filter(None, [os.environ.get('GZ_SIM_RESOURCE_PATH'), os.path.dirname(share)]))
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=os.path.join(share, 'worlds', 'rl_square_arena.sdf')),
        DeclareLaunchArgument('world_name', default_value='rl_square_arena'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'rsp.launch.py')),
            launch_arguments={'use_sim_time': use_sim_time, 'use_ros2_control': 'false'}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gz_share, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': ['-r -v 3 ', world], 'on_exit_shutdown': 'true'}.items()),
        Node(package='ros_gz_sim', executable='create',
             arguments=['-topic', 'robot_description', '-name', 'hv_bot', '-x', '0', '-y', '-6', '-z', '0.2'],
             parameters=[{'use_sim_time': use_sim_time}], output='screen'),
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             parameters=[{'config_file': os.path.join(share, 'config', 'gz_bridge.yaml'), 'use_sim_time': use_sim_time}],
             output='screen'),
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             arguments=[['/world/', world_name, '/set_pose@ros_gz_interfaces/srv/SetEntityPose'],
                        ['/world/', world_name, '/control@ros_gz_interfaces/srv/ControlWorld']],
             output='screen'),
        Node(package='rviz2', executable='rviz2', condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', os.path.join(share, 'rviz', 'original.rviz')],
             parameters=[{'use_sim_time': use_sim_time}], output='screen'),
    ])
