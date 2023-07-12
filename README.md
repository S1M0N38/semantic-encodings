# semantic-encodings

A simple notebook explaing the core concepts in
**Beyond One-Hot-Encoding: Injecting Semantics to Drive Image Classifiers**

## Installation

We assume that you have Python 3.10 and jupyer-lab/jupyter-notebook installed
on your system.

1. Clone this repository: `git clone https://github.com/S1M0N38/semantic-encodings.git`
2. Move inside the repository: `cd semantic-encodings`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment: `source venv/bin/activate`
5. Install the dependencies: `python -m pip install -r requirements.txt`
6. Install the kernel: `python -m ipykernel install --user --name=semantic-encodings`

## Usage

We assume that you follow the installation instructions above.

1. Open the notebook: `jupyter-lab semantec-encodings.ipynb` or
   `jupyter-notebook semantec-encodings.ipynb`
2. Select the kernel: `Kernel -> Change kernel -> semantic-encodings`

## Uninstall

If you need to uninstall this dir and all libraries installed in the virtual
environment, simply remove `semantic-encodings` (If you have followed the
installation the virtual env directory is inside `semantic-encodings`).

More over you need to remove the kernel installed. Run the command `python -m
ipykernel install --user --name=semantic-encodings` again and then delete
the directory that appears in the output.
