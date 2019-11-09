from setuptools import setup, find_packages


setup(
    name='rgb-depth-dataset-tools',
    packages=find_packages(),
    zip_safe=True,
    install_requires=[
        'numpy>=1.7.3,<2',
        'matplotlib>=3.1.1,<4',
        'seaborn>=0.9,<1',
        'tqdm>=4.38.0,<5',
        'torch>=1.3.1,<2',
        'pyro-ppl>=0.5.1,<1',
    ],
)
