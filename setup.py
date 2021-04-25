from setuptools import setup, find_packages

# Note that this version may be untrue
setup(name='godr',
      version='1.0',
      packages=find_packages(where='src'),
      package_data={
          'godr.frontend.ui': ['*'],
      },
      package_dir={'': 'src'},
      python_requires='>3.6',
      install_requires=[
          'matplotlib',
          'opencv-python-headless',
          'sgfmill',
          'pyqt5',
          'numpy',
          'QtAwesome',
          'PyMuPDF',
          'pagerange'
      ],
      entry_points={
          'console_scripts': [
              'godr=godr.frontend.app_ui:main'
          ]
      })
