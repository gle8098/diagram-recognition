from setuptools import setup, find_packages

# Note that this version may be untrue
setup(name='godr',
      version='1.1',
      packages=find_packages(where='src'),
      package_data={
          'godr.frontend.ui': ['*'],
      },
      package_dir={'': 'src'},
      python_requires='>3.6',
      install_requires=[
          'opencv-python-headless~=4.5.3',
          'sgfmill==1.1.1',
          'pyqt5==5.15.4',
          'numpy',
          'QtAwesome',
          'PyMuPDF==1.20.2',
          'pagerange',
          'pathvalidate',
          'onnxruntime==1.12.1'
      ],
      entry_points={
          'console_scripts': [
              'godr=godr.frontend.app_ui:main',
              'sgf_joiner=godr.sgf_joiner:main',
              'sgf_joiner_ui=godr.frontend.sgf_joiner_ui:main'
          ]
      })
