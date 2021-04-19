# Copyright 2021 The BigBird Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install BigBird."""

import setuptools

# Get install requirements from the REQUIREMENTS file.
with open('requirements.txt') as fp:
  _REQUIREMENTS = fp.read().splitlines()

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setuptools.setup(
    name='bigbird',
    version='0.0.1',
    description='Big Bird: Transformers for Long Sequences',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/google-research/bigbird',
    license='Apache 2.0',
    packages=[
        'bigbird', 'bigbird.core', 'bigbird.classifier',
        'bigbird.pretrain', 'bigbird.summarization'
    ],
    package_data={'bigbird': ['vocab/*']},
    scripts=[],
    install_requires=_REQUIREMENTS,
    keywords='deeplearning machinelearning nlp classifier qa summarization transformer pretraining',
)
