from setuptools import setup, find_packages

setup(
    name='sds_playground',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # list your package dependencies here
        # e.g., 'requests>=2.23.0',
    ],
    # Additional metadata
    author='Yannik Frisch',
    author_email='yannik.frisch@gris.tu-darmstadt.de',
    description='Surgical Data Science Playground',
    url='https://github.com/YFrisch/SDS_Playground',
)