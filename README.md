# semantic-encodings

A simple notebook explaining the core concepts in
**Beyond One-Hot-Encoding: Injecting Semantics to Drive Image Classifiers**

## Installation

We assume that you have Python 3.10 and jupyter-lab/jupyter-notebook installed
on your system.

1. Clone this repository: `git clone https://github.com/S1M0N38/semantic-encodings.git`
1. Move inside the repository: `cd semantic-encodings`
1. Create a virtual environment: `python -m venv venv`
1. Activate the virtual environment: `source venv/bin/activate`
1. Install the dependencies: `python -m pip install -r requirements.txt`
1. Install the kernel: `python -m ipykernel install --user --name=semantic-encodings`

## Usage

We assume that you have followed the installation instructions above.

1. Open the notebook: `jupyter-lab semantic-encodings.ipynb` or
   `jupyter-notebook semantic-encodings.ipynb`
1. Select the kernel: `Kernel -> Change kernel -> semantic-encodings`

## Uninstall

If you need to uninstall this directory and all libraries installed in the virtual
environment, simply remove the `semantic-encodings` directory (If you have followed the
installation, the virtual environment directory is inside `semantic-encodings`).

Moreover, you need to remove the kernel installed. Run the command `jupyter kernelspec uninstall semantic-encodings` to remove the kernel.
