from setuptools import setup
from os import path

requirements = [
    'matplotlib>=3.3.0',
    'numpy>=1.10',
    'pandas',
    'python-dateutil',
    'scipy>=0.18',
    'seaborn',  # only used by bnse.py
    'torch',
]

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='mogptk',
      version='0.2.6',
      description='Multi-Output Gaussian Process ToolKit',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/GAMES-UChile/mogptk',
      author='Taco de Wolff, Alejandro Cuevas, Felipe Tobar',
      author_email='tacodewolff@gmail.com',
      license='MIT',
      packages=['mogptk', 'mogptk.gpr', 'mogptk.models'],
      keywords=['MOGP', 'MOSM', 'GP', 'Gaussian Process', 'Multi-Output', 'Tobar', 'Parra'],
      python_requires='>=3.6',
      install_requires=requirements,
)
