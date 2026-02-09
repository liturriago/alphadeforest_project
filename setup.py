from setuptools import setup, find_packages

setup(
    name="alphadeforest",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "numpy",
        "webdataset",
        "pydantic",
        "pyyaml",
        "tqdm",
        "matplotlib",
        "kagglehub",
    ],
    extras_require={
        "test": ["pytest"],
    },
    author="Tu Nombre",
    description="Framework para detección de deforestación usando CAE y Memory Networks",
)