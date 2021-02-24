from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='torchopenl3',
    description='',
    version='0.0.1',
    author='Hugo Flores Garcia',
    author_email='hf01049@georgiasouthern.edu',
    url='https://github.com/hugofloresgarcia/torchopenl3',
    install_requires=[
        'pytorch-lightning', 
        'numpy', 'torch', 'librosa', 
        'audio_utils @ git+https://github.com/hugofloresgarcia/audio-utils'
    ],
    packages=['torchopenl3'],
    package_data={'torchopenl3': ['torchopenl3/assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
