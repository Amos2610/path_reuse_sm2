from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    params_file = PathJoinSubstitution([
        FindPackageShare('path_reuse_sm2'),
        'io', 'params.yaml'
    ])

    prsm = Node(
        package='path_reuse_sm2',
        executable='sm_node.py',
        name='prsm_node',
        output='screen',
        parameters=[{
            'flow': ['SkillGraspObj', 'SkillUpdatePathSeed', 'SkillPutObj', 'SkillUpdatePathSeed'],
            'loop': True,               # flowの最後→最初に戻る
            'hold_after': True,           # 実行後もしばらくノードを生かす
            'params_file': params_file,   # いまはログのみ
        }],
    )

    viewer = Node(
        package='yasmin_viewer',
        executable='yasmin_viewer_node',
        name='yasmin_viewer',
        output='screen',
        # parameters=[{'port': 8080}],  # 必要なら
    )

    return LaunchDescription([viewer, prsm])