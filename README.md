# MultArtRec
This repo is the implementation of paper **MultArtRec: A Multi-Modal Neural Topic Modeling for Integrating Image and Text Features in Artwork Recommendation**. It is a module added to **[cornac](https://github.com/PreferredAI/cornac)**.

# Get Started

### Prepare Conda Environemnt.

Create new conda environment with Python 3.9.0. (Conda should be installed at first.)
  ```bash
  conda activate cornac python==3.9.0
  ```
Activate conda environment.
  ```bash
  conda activate cornac
  ```
Install Cornac. (Here we test Cornac 1.16.)
  ```bash
  pip3 install cornac
  ```
Install TensorFlow. (Here we test TensorFlow 2.13.0.)
  ```bash
  pip install tensorflow
  ```

### Copy Folder

Copy the folder `mar` to `cornac/models/` where the Cornac package is installed. For example: `anaconda/envs/cornac/lib/python3.9/site-packages/`.

### Edit File

In `cornac/__init__.py`,  
Add `from .mar import MAR`.

### Run Code
  ```bash
  python text_modal_mar.py
  ```


