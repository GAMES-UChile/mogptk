from setuptools import setup
from os import path

requirements = [
    'numpy>=1.10',
    'pytorch',
    'matplotlib>=3.3.0',
    'scipy>=0.18',
    'pandas',
    'python-dateutil',
    'seaborn',  # only used by bnse.py
    'scikit-learn',  # only used by errors.py
]

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='mogptk',
      version='0.1.10',
      description='Multi-Output Gaussian Process ToolKit',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/GAMES-UChile/MultiOutputGP-Toolbox',
      author='Taco de Wolff, Alejandro Cuevas, Felipe Tobar',
      author_email='tacodewolff@gmail.com',
      license='MIT',
      packages=['mogptk', 'mogptk.kernels'],
      keywords=['MOGP', 'MOSM', 'GP', 'Gaussian Process', 'Multi-Output', 'Tobar', 'Parra'],
      python_requires='>=3.6',
      install_requires=requirements,
)
