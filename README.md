# AffiGrapher: Contrastive Heterogeneous Graph Learning with Aromatic Virtual Nodes for RNA-Small Molecule Binding Affinity Prediction


Implementation of AfiiGrapher, by Junkai Wang.


## Setup Environment

We recommend setting up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).

This is an example for how to set up a working conda environment to run the code (but make sure to use the correct pytorch, DGL, cuda versions or cpu only versions):

   `conda create --name AffiGrapher python=3.9`
   `conda activate AffiGrapher`

   **The relevant environment can be successfully installed by executing the following commands in sequence:**

   `conda install pytorch==2.3.1 cudatoolkit=11.8 -c pytorch`

   `conda install -c dglteam/label/th23_cu118 dgl`

   `conda install -c conda-forge rdkit`

   `conda install -c conda-forge biopython`

   `conda install -c conda-forge scikit-learn`

   `conda install -c conda-forge prolif`

   `pip install prefetch-generator`

   `pip install lmdb`

   `pip install numpy==1.24.3`


## **Retraining and Testing AffiGrapher **

1. you need to download the traing dataset and the lmdb data can be accessed via “https://drive.google.com/file/d/1r17CVjMgq-qYVtbDLx1yuAIRpMWCKdW5/view?usp=drive_link”. 
   (You can also use your own private data, As long as it can fit to EquiScore after processing)
2. use uniprot id to 10-fold split data in `AffiGrapher/workdir/constrastive/2025-01-12-16-36-33/splits.json`
3. run Train_contrastive_aff.py script for reTraining:
   `python Train_contrastive_aff.py --ligandonly`、
4. run Train_contrastive_aff.py script for testing:
   `python Train_contrastive_aff.py --ligandonly --onlytest`

## Citation

waiting accepted
## License

MIT
