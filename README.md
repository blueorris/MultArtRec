# MultArtRec
This repo is the implementation of paper **MultArtRec: A Multi-Modal Neural Topic Modeling for Integrating Image and Text Features in Artwork Recommendation**. It is a module added to **[Cornac framework](https://github.com/PreferredAI/cornac)**.

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

Copy the folder `mar/` to `cornac/models/` where the Cornac package is installed. For example: `anaconda/envs/cornac/lib/python3.9/site-packages/`.

### Edit File

In `cornac/__init__.py`, add `from .mar import MAR`.

### Download Data

For convenience, we have prepared the required data in advance.  
Download all of them and save to the `data/` folder.

| File Name     | Download Link |
| ----------- | ----------- |
| `feedback.npy`      | [Google Drive Link](https://drive.google.com/file/d/1ct2VbClkNNpvQxs_ekMdVMaw5bfaaxoZ/view?usp=drive_link)       |
| `text_features_bert_base.npy`   | [Google Drive Link](https://drive.google.com/file/d/14_4YdbCSg5XUzPfa9FLtQz3xsQfvpNu7/view?usp=drive_link)        |
|`item_ids.npy`  |[Google Drive Link](https://drive.google.com/file/d/16PfxjpiFfBn4_n_j85X0htQj-ePSe9m5/view?usp=drive_link)|

### Run Code
  ```bash
  python mar_exp.py
  ```


