# semantic-encodings

**Beyond One-Hot-Encoding: Injecting Semantics to Drive Image Classifiers**
\[[Conference Paper](https://link.springer.com/chapter/10.1007/978-3-031-44067-0_27)\] | \[[arXiv Version](http://arxiv.org/abs/2308.00607v1)\]

*This is not the code used in the paper but a simple notebook to explain the core concepts.*

## Installation

We assume that you have Python 3.10 and jupyter-lab/jupyter-notebook installed on your system.

1. Clone this repository: `git clone https://github.com/S1M0N38/semantic-encodings.git`
1. Move inside the repository: `cd semantic-encodings`
1. Create a virtual environment: `python -m venv venv`
1. Activate the virtual environment: `source venv/bin/activate`
1. Install the dependencies: `python -m pip install -r requirements.txt`
1. Install the kernel: `python -m ipykernel install --user --name=semantic-encodings`

## Usage

We assume that you have followed the installation instructions above.

1. Open the notebook: `jupyter-lab semantic-encodings.ipynb` or `jupyter-notebook semantic-encodings.ipynb`
1. Select the kernel: `Kernel -> Change kernel -> semantic-encodings`

## Uninstall

If you need to uninstall this directory and all libraries installed in the virtual environment, simply remove the `semantic-encodings` directory (If you have followed the installation, the virtual environment directory is inside `semantic-encodings`).

Moreover, you need to remove the kernel installed. Run the command `jupyter kernelspec uninstall semantic-encodings` to remove the kernel.

## Cite

```bibtex
@article{SemanticsEncPerott2023,
  author = {Perotti, Alan and Bertolotto, Simone and Pastor, Eliana and Panisson, Andr\'{e}},
  eprint = {2308.00607v1},
  month = {8},
  primaryclass = {cs.cv},
  title = {Beyond One-Hot-Encoding: Injecting Semantics to Drive Image Classifiers},
  url = {http://arxiv.org/abs/2308.00607v1},
  year = {2023},
}
```
