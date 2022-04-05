import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()


setuptools.setup(
    name="disease-codification",
    version="0.1.6",
    author="Jose Barros",
    author_email="jose.barros.s@ug.uchile.cl",
    description="Implements a model for disease codificacion using a Extreme Classification Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/josebarross/disease-codification",
    project_urls={
        "Bug Tracker": "https://github.com/josebarross/disease-codification/issues",
    },
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=install_requires,
    include_package_data=True,
)
