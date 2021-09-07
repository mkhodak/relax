from setuptools import setup


setup(name='relax',
      url='https://github.com/mkhodak/relax',
      author='Misha Khodak',
      author_email='khodak@cmu.edu',
      packages=['relax'],
      install_requires=['torch'],
      version='0.0.0',
      license='MIT',
      description='NAS relaxation tools',
      long_description=open('README.md').read(),
      )
