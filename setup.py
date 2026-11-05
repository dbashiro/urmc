from glob import glob

from setuptools import setup

package_name = 'robothub'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Equipe RobotHub',
    maintainer_email='mathieu.brasseur9@orange.fr',
    description='Missions TurtleBot3 Burger - URMRC 2026',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'track_mission = robothub.track_mission:main',
            'enclosure_mission = robothub.enclosure_mission:main',
            'storage_mission = robothub.storage_mission:main',
            'electromagnet = robothub.electromagnet_node:main',
            'hsv_tuner = robothub.hsv_tuner:main',
            'capture_templates = robothub.capture_templates:main',
            'camera_web = robothub.camera_web:main',
            'control_panel = robothub.control_panel:main',
        ],
    },
)
