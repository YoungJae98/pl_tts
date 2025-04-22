from setuptools import find_packages, setup


def get_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()

    return requirements


setup(
    name="vits-pl",
    version="0.0.1",
    author="yj",
    package_dir={"": "src"},
    packages=find_packages(include = ["src", "src.*"]),
    install_requires=get_requirements(),
    python_requires='>=3.9',
)