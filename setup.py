#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup
setup(
    name='StarmapSpots',
    version='0.1',
    py_modules=['StarmapSpots'],
    install_requires=[
        'Click', 'starfish'
    ],
    entry_points='''
        [console_scripts]
        StarmapSpots=StarmapSpots:cli
    ''',
)

