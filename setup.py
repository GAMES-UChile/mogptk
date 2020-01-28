import sys
from pkg_resources import parse_version
from setuptools import setup, find_packages

is_py37 = sys.version_info.major == 3 and sys.version_info.minor == 7

requirements = [
    'numpy>=1.18',
    'gpflow @ git+https://github.com/GPflow/GPflow.git@develop#egg=gpflow',
    'tensorflow-probability>=0.9',
    'matplotlib',
    'scipy',
    'scikit-learn',
    'pandas',
    'seaborn',
    'python-dateutil',
    'ipython',
    'dataclasses',
] + ([] if is_py37 else ["dataclasses"])

try:
    import tensorflow as tf
    if parse_version(tf.__version__) < parse_version('2.1.0'):
        raise DeprecationWarning("TensorFlow version below minimum requirement")
except (ImportError, DeprecationWarning):
    requirements.append('tensorflow~=2.1')

setup(name='mogptk',
      version='0.1',
      description='Multi-Output Gaussian Process ToolKit',
      url='https://github.com/GAMES-UChile/MultiOutputGP-Toolbox',
      author='Taco de Wolff, Alejandro Cuevas, Felipe Tobar',
      author_email='tacodewolff@gmail.com',
      license='MIT',
      packages=find_packages('.', exclude=['examples']),
      include_package_data=False,
      keywords=['MOGP', 'MOSM', 'GP', 'Gaussian Process', 'Multi-Output', 'Tobar', 'Parra'],
      python_requires='>=3.6',
      extra_require={'TensorFlow with GPU': 'tensorflow-gpu'},
      install_requires=requirements,
)
