# CytoVI

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/florianingelfinger/CytoVI/test.yaml?branch=main
[link-tests]: https://github.com/florianingelfinger/CytoVI/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/CytoVI

Variational inference for antibody-based single cell technologies. 

CytoVI accomplishes the following analysis tasks:

-   Integration/batch correction
-   Differential protein expression analysis
-   Label-free differential abundance analysis
-   Imputation of unseen proteins from overlapping antibody panels
-   Technology integration (e.g. Flow and Mass cytometry)
-   RNA/modality imputation after integration of flow/mass cytometry data with CITE-seq data
-   Automated cell annotation via transfer learning

## Getting started

To get started please check out the basic analysis notebook in the [docs](https://github.com/florianingelfinger/CytoVI/blob/main/docs/notebooks/Basic_CytoVI_workflow.ipynb).

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install CytoVI:

<!--
1) Install the latest release of `CytoVI` from `PyPI <https://pypi.org/project/CytoVI/>`_:

```bash
pip install CytoVI
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/florianingelfinger/CytoVI.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/florianingelfinger/CytoVI/issues
[changelog]: https://CytoVI.readthedocs.io/latest/changelog.html
[link-docs]: https://CytoVI.readthedocs.io
[link-api]: https://CytoVI.readthedocs.io/latest/api.html
