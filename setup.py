from setuptools import setup, find_packages

setup(
    name="funcrl",
    version='0.1.0',
    packages=[package for package in find_packages(
    ) if package.startswith("funcrl")],
    package_data={"funcrl": ["py.typed", "version.txt"]},
    description="Description",
    author="Aiden Lee",
    url="https://github.com/func25/funcrl",
    author_email="phuongle0205@gmail.com",
    keywords="funcrl",
    license="MIT",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        'tqdm',
        'matplotlib',
        'gymnasium',
        'numpy',
    ],
)
