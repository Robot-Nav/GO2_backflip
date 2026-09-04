"""Editable installation for the PPO-backflip task."""

from setuptools import find_packages, setup


setup(
    name="ppo-backflip",
    version="0.1.0",
    description="Phase-conditioned PPO backflip control for the Unitree Go2 quadruped",
    packages=find_packages(include=("isaaclab_backflip", "isaaclab_backflip.*")),
    python_requires=">=3.10",
    zip_safe=False,
)

