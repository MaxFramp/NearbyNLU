from setuptools import setup, find_packages

setup(
    name="nearbynlu",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "googlemaps",
        "tensorflow",
        "numpy",
    ],
) 