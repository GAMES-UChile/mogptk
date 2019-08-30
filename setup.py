from setuptools import setup

setup(name='mogptk',
      version='0.1',
      description='Multi-Output Gaussian Process ToolKit',
      url='https://github.com/GAMES-UChile/MultiOutputGP-Toolbox',
      author='Taco de Wolff',
      author_email='tacodewolff@gmail.com',
      license='MIT',
      packages=['mogptk'],
      keywords=['MOGP', 'MOSM', 'GP', 'Gaussian Process', 'Multi-Output', 'Tobar', 'Parra'],
      install_requires=[
          'numpy',
      ],
)
