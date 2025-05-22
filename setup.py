from setuptools import find_packages, setup


def get_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()

    return requirements


setup(
    name="pl_tts",
    version="1.0.0",
    author="genius_98",
    package_dir={"": "src"},
    packages=find_packages(include = ["src", "src.*"]),
    install_requires=get_requirements(),
    python_requires='>=3.9',
)