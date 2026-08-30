from glob import glob

from setuptools import find_packages, setup

package_name = 'nav_learning'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='harivathsha',
    maintainer_email='harivathsharamesh@gmail.com',
    description='ROS 2 environment and from-scratch PPO navigation training',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'train_ppo = nav_learning.train_ppo:main',
            'evaluate_ppo = nav_learning.evaluate_ppo:main',
            'plot_ppo_results = nav_learning.plot_ppo_results:main',
        ],
    },
)
