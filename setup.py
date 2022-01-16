from setuptools import setup

setup(
    name = 'crowdsourcephoto',
    packages = ['crowdsource'],
    version = '0.5.6',
    description = 'Crowded field photometry pipeline',
    author = 'Andrew Saydjari',
    author_email = 'aksaydjari@gmail.com',
    url = 'https://github.com/schlafly/crowdsource',
    download_url = 'https://github.com/schlafly/crowdsource/archive/refs/tags/v0.5.6.tar.gz',
    license = 'MIT',
    install_requires=[
        'astropy',
        'numpy',
        'scipy >= 1.5',
        'scikit-image',
        'guppy3',
        'matplotlib',
        'keras',
        'tensorflow >= 2'
    ]
)
