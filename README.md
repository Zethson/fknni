# fknni

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/zethson/fknni/test-cpu.yaml?branch=main
[link-tests]: https://github.com/zethson/fknni/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/fknni

Fast implementations of KNN imputation using faiss.
Might support more backends such as annoy and GPUs in the future.
Help is more than welcome!

![image](https://github.com/Zethson/fknni/assets/21954664/216cb745-e0c3-481b-9cae-59bc9a77cdfe)

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

- [API documentation][link-api].
- [Faiss example notebook][link-faiss-example].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

Install the latest release of `fknni` from [PyPI](https://pypi.org/project/fknni):

```bash
pip install fknni
```

Install the latest development version:

```bash
pip install git+https://github.com/zethson/fknni.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

[issue-tracker]: https://github.com/zethson/fknni/issues
[changelog]: https://fknni.readthedocs.io/latest/changelog.html
[link-docs]: https://fknni.readthedocs.io
[link-api]: https://fknni.readthedocs.io/latest/api.html
[link-faiss-example]: https://fknni.readthedocs.io/en/latest/notebooks/faiss.html
