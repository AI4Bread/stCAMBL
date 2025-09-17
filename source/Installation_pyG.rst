.. S3RL documentation master file, created by
   sphinx-quickstart on Wed Apr 16 19:43:51 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation (pyG framework)
============

Software Dependencies
---------------------
.. code-block:: python

   h5py==3.8.0
   louvain==0.8.2
   MarkupSafe==2.1.5
   matplotlib==3.5.3
   matplotlib-inline==0.1.6
   munkres==1.1.4
   networkx==2.6.3
   numpy
   opencv-python==4.8.1.78
   pandas==1.3.5
   Pillow==9.3.0
   PyYAML
   rpy2==3.5.14
   scanpy==1.9.3
   scikit-learn
   scikit-misc==0.1.4
   scipy
   seaborn==0.12.2
   timm==0.9.10
   torch==1.13.0+cu117
   torch-geometric
   torchaudio==0.13.0+cu117
   torchvision==0.14.0+cu117
   umap-learn==0.5.4

Installation
------------

Download the S3RL code from GitHub: https://github.com/AI4Bread/S3RL, clone the repository with the following command:

.. code-block:: bash

   git clone git@github.com:AI4Bread/S3RL.git
   cd S3RL

Then, create a new conda environment and install the required packages:

.. code-block:: bash
   conda create -n S3RL python=3.7.12
   conda activate S3RL
   pip install -r requirements.txt


.. note::

The code is tested with Python 3.7.12 and PyTorch 1.13.0 on a single NVIDIA GeForce RTX 3090 GPU.
If you encounter any issues, please check the compatibility of the packages in requirements.txt with your Python version.
Additionally, different versions of libraries and different GPU devices may lead to varied outcomes,
so to reproduce our results, please use the same versions and hardware configuration as specified.

Dataset Directory Structure

Download the datasets and place them in the Data directory, ensuring the directory structure appears as follows:

.. code-block:: text

   S3RL
   ├── Data
   │   ├── DLPFC
   │   │   ├── 151673
   │   │   ├── 151674
   │   ├── Nanostring
   │   ├── Human_Breast_Cancer
   │   ├── Mouse_Brain_Anterior

Processed datasets can be downloaded from the following links:

- `DLPFC <https://github.com/LieberInstitute/spatialLIBD?tab=readme-ov-file#raw-data>`__
- `Nanostring <https://purdue0-my.sharepoint.com/personal/tang385_purdue_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Ftang385%5Fpurdue%5Fedu%2FDocuments%2FSigra%5Fdataset&ga=1>`__
- `Human Breast Cancer <https://www.10xgenomics.com/resources/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0>`__
- `Mouse Brain Anterior <https://drive.google.com/drive/folders/1jDmx8IjiGhOD__spuuhFB1fWVDJtv5CU>`__
- `Human Lung Cancer <https://drive.google.com/drive/folders/14ZHjGeyyVUnYn_0EqpHmNVo1V1F5B1Hx>`__
- `Mouse Hippocampus Starmap <https://drive.google.com/drive/folders/14ZHjGeyyVUnYn_0EqpHmNVo1V1F5B1Hx>`__
- `Soybean <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270392>`__
Running the Model

To start training, run the following command:

.. code-block:: bash

   bash demo_train.sh