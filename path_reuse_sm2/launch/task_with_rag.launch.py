import math
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    default_params = PathJoinSubstitution([
        FindPackageShare('path_reuse_sm2'),
        'io', 'params.yaml'
    ])

    declare_params = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='YAML file with node parameters'
    )
    declare_use_viewer = DeclareLaunchArgument(
        'use_viewer',
        default_value='false',
        description='Launch yasmin_viewer node'
    )

    params_file = LaunchConfiguration('params_file')

    # hand_camera_link は RealSense ドライバが内部フレームを配信するが、
    # link_eef → hand_camera_link のマウント変換はドライバが配信しない。
    # URDF (realsense_d435i.urdf.xacro) の hand_camera_link_joint 定義に従い静的TFを補完する。
    # simulate_only モードで TF ツリーが分断されるのを防ぐ。
    hand_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='hand_camera_link_tf',
        output='screen',
        arguments=[
            '--x', '0.06746',
            '--y', '-0.0175',
            '--z', '0.0237',
            '--roll', str(math.pi),
            '--pitch', str(-math.pi / 2),
            '--yaw', '0',
            '--frame-id', 'link_eef',
            '--child-frame-id', 'hand_camera_link',
        ],
    )

    prsm = Node(
        package='path_reuse_sm2',
        executable='sm_node_with_rag.py',
        name='prsm_node',
        output='screen',
        parameters=[
            params_file,
            {
                'loop': False,               # flowの最後→最初に戻る
                'hold_after': True,           # 実行後もしばらくノードを生かす
            }
        ],
    )

    viewer = Node(
        package='yasmin_viewer',
        executable='yasmin_viewer_node',
        name='yasmin_viewer',
        output='screen',
        parameters=[{'port': 8001}],
        condition=IfCondition(LaunchConfiguration('use_viewer')),
    )

    return LaunchDescription([
        declare_params,
        declare_use_viewer,
        hand_camera_tf,
        viewer,
        prsm,
    ])
