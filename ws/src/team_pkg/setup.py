import os
from glob import glob
from setuptools import setup

package_name = "team_pkg"
setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "hello_node = team_pkg.hello_node:main",
        "ant1_drive = team_pkg.ant1_drive:main",
        "ant1_keys = team_pkg.ant1_keys:main",
        "ant1_watchdog = team_pkg.ant1_watchdog:main",
        "ant1_goto = team_pkg.ant1_goto:main",
        "ant1_nav_speed = team_pkg.ant1_nav_speed:main",
        "cf_guard = team_pkg.cf_guard:main",
        "cf_keys = team_pkg.cf_keys:main",
        "ant_link = team_pkg.ant_link:main",
        "mission_calib = team_pkg.mission_calib:main",
        "mission_follow = team_pkg.mission_follow:main",
        "mission_scout = team_pkg.mission_scout:main",
        "mission_hop = team_pkg.mission_hop:main",
        "vision_link = team_pkg.vision_link:main",
        "vision_calib = team_pkg.vision_calib:main",
        "mission_rescue = team_pkg.mission_rescue:main",
        "ant1_bridge = team_pkg.ant1_bridge:main",
        "ant1_dummy_pub = team_pkg.ant1_dummy_pub:main",
    ]},
)
