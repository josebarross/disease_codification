import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()


setuptools.setup(
    name="dac-divide-and-conquer",
    version="1.3.2",
    author="Jose Barros",
    author_email="jose.barros.s@ug.uchile.cl",
    description="Implements an architecture for extreme multi-label classification leveraging semantic relations between labels. Extensively tested for disease coding.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/josebarross/dac-divide-and-conquer",
    project_urls={
        "Bug Tracker": "https://github.com/josebarross/dac-divide-and-conquer/issues",
    },
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=install_requires,
    include_package_data=True,
)
