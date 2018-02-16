from __future__ import absolute_import
from setuptools import setup, find_packages

setup(name=u'baselines',
      packages=[package for package in find_packages()
                if package.startswith(u'baselines')],
      install_requires=[
          u'gym',
          u'scipy',
          u'tqdm',
          u'joblib',
          u'zmq',
          u'dill',
          u'azure==1.0.3',
          u'progressbar2',
          u'mpi4py',
          u'cloudpickle',
      ],
      description=u"OpenAI baselines: high quality implementations of reinforcement learning algorithms",
      author=u"OpenAI",
      url=u'https://github.com/openai/baselines',
      author_email=u"gym@openai.com",
      version=u"0.1.4")
