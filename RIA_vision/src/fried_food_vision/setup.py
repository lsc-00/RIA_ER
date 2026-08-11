from setuptools import find_packages, setup


package_name = 'fried_food_vision'


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(
        exclude=['test']
    ),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='edlsc12',
    maintainer_email='edlsc12@example.com',
    description=(
        'ROS 2 fried food detection and '
        'image transport performance comparison package'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_rate_monitor = fried_food_vision.image_rate_monitor:main',
            'yolo_detection_node = fried_food_vision.yolo_detection_node:main',

            # YOLO 중심 Pixel + aligned depth
            # → Camera XYZ 좌표 계산
            'depth_position_node = fried_food_vision.depth_position_node:main',
        ],
    },
)