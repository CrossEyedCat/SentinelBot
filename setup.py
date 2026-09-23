import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'sentinel_patrol'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.npz') + glob('config/*.pt')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf') + glob('worlds/*.map')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'models', 'sentinel_mk2'), glob('models/sentinel_mk2/model.*')),
        (os.path.join('share', package_name, 'models', 'sentinel_mk2', 'meshes'), glob('models/sentinel_mk2/meshes/*.stl')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Artem Zenkevich',
    maintainer_email='110236400+CrossEyedCat@users.noreply.github.com',
    description='FSM-based autonomous patrol robot for ROS 2 (TurtleBot3 or SentinelBot Mk II in Gazebo).',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'patrol_fsm = sentinel_patrol.patrol_fsm_node:main',
            'scan_to_range = sentinel_patrol.scan_to_range_node:main',
            'mecanum_base_sim = sentinel_patrol.mecanum_base_sim_node:main',
            'occupancy_mapper = sentinel_patrol.occupancy_mapper_node:main',
            'path_planner = sentinel_patrol.path_planner_node:main',
            'column_tour = sentinel_patrol.column_tour_node:main',
        ],
    },
)
