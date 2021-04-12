import setuptools

setuptools.setup(
    name='hep_anomaly',
    description='HEP-AnomalyDetection',
    install_requires=[
        'pytorch-lightning', 'numpy', 'sklearn', 'tqdm', 'matplotlib'
    ]
)