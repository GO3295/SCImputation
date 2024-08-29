from setuptools import setup

setup(name='afMF',
      version='0.1',
      description='afMF:single cell RNA-seq dropout imputation',
      url='http://github.com/jh50/scRNAseqImpute',
      author='AC & JH',
      author_email='1155186461@link.cuhk.edu.hk',
      license='MIT',
      packages=['afMF'],
      install_requires=[
          'pandas',
          'numpy',
          'scikit-learn',
      ],
      zip_safe=False)
