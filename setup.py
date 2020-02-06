import sys
from os import path
from pkg_resources import parse_version
from setuptools import setup, find_packages

is_py37 = sys.version_info.major == 3 and sys.version_info.minor == 7

requirements = [
    'numpy>=1.10',
    'gpflow==2.0.0rc1',
    'matplotlib',
    'scipy>=0.18',
    'scikit-learn',
    'pandas',
    'seaborn',
    'python-dateutil',
    'ipython',
    'dataclasses',
] + ([] if is_py37 else ["dataclasses"])

try:
    import tensorflow as tf
    if parse_version(tf.__version__) < parse_version('2.0.0'):
        raise DeprecationWarning("TensorFlow version below minimum requirement")
    if parse_version(tf.__version__) < parse_version('2.1.0'):
        requirements.append('tensorflow-probability==0.8')
    else:
        requirements.append('tensorflow-probability==0.9')
except (ImportError, DeprecationWarning):
    if is_py37:
        requirements.append('tensorflow==2.1')
        requirements.append('tensorflow-probability==0.9')
    else:
        requirements.append('tensorflow==2.0')
        requirements.append('tensorflow-probability==0.8')

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='mogptk',
      version='0.1.3',
      description='Multi-Output Gaussian Process ToolKit',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/GAMES-UChile/MultiOutputGP-Toolbox',
      author='Taco de Wolff, Alejandro Cuevas, Felipe Tobar',
      author_email='tacodewolff@gmail.com',
      license='MIT',
      packages=['mogptk'],
      keywords=['MOGP', 'MOSM', 'GP', 'Gaussian Process', 'Multi-Output', 'Tobar', 'Parra'],
      python_requires='>=3.6',
      extra_require={'TensorFlow with GPU': 'tensorflow-gpu'},
      install_requires=requirements,
)
