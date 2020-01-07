from setuptools import setup

setup(name='mogptk',
      version='0.1',
      description='Multi-Output Gaussian Process ToolKit',
      url='https://github.com/GAMES-UChile/MultiOutputGP-Toolbox',
      author='Taco de Wolff, Alejandro Cuevas, Felipe Tobar',
      author_email='tacodewolff@gmail.com',
      license='MIT',
      packages=['mogptk'],
      keywords=['MOGP', 'MOSM', 'GP', 'Gaussian Process', 'Multi-Output', 'Tobar', 'Parra'],
      python_requires='~=3.7.0',
      install_requires=[
          'numpy~=1.18.0',
          'gpflow~=2.0.0rc1',
          'tensorflow~=2.0.0',
          'matplotlib',
          'scipy',
          'scikit-learn',
          'pandas',
          'seaborn',
          'python-dateutil',
      ],
)
