from setuptools import setup, find_packages

setup(
    name="nearbynlu",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "python-dotenv>=1.0.0",
        "googlemaps>=4.10.0",
        "tensorflow>=2.12.0",
        "numpy>=1.24.0",
        "sentence-transformers>=4.1.0",
        "httpx>=0.24.0",
        "starlette>=0.27.0",
    ],
) 