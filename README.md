# DrugPairScoringDR

[arxiv-image]: https://img.shields.io/badge/ArXiv-2209.09383-orange.svg
[arxiv-url]: https://arxiv.org/abs/2209.09383

[![Arxiv][arxiv-image]][arxiv-url]

This is the source code repository for "Distributed representations of graphs for drug pair scoring". It also contains the distributed representations of molecular graphs learned over the drug sets in DrugCombDB, DrugComb, DrugbankDDI and TwoSides datasets.

## Using our trained distributed representations of graphs

You can find our learned distributed representations in `data/embeddings`. Each of the embeddings are saved in `.json` files which contain a mapping from the drug to its distributed representation within the set. 

Our naming convention is: `<Dataset>_<SubstructurePattern>_<EmbeddingDimensionality>_<NumEpochsTrained>.json`

## Prerequisites and installation

There are several prerequisites to using our code, chief amongst them a modified version of ChemicalX called `modded_chemicalx` we include in this repository (all credit to original ChemicalX authors). This contains our DR augmented model implementations and augmented data loaders utilising the distributed representations. We include concrete installation instructions below. References and citation suggestions for these packages (and our project) can be found below. 

The ChemicalX and TorchDrug v.0.1.2 require Python 3.8 which we recommend being installed in a conda virtual environment

```bash
conda create -n drvenv python=3.8
conda activate drvenv
```

We can install the required packages

### CPU

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install torchdrug==0.1.2
```

Do a local installation of the modded_chemicalx package, followed by the Geo2DR package.

```bash
cd modded_chemicalx
pip install -e .
cd ..
pip install git+https://github.com/paulmorio/geo2dr.git
```

### GPU

If you have a GPU you can follow these instructions

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
pip install torchdrug==0.1.2
```

Do a local installation of the modded_chemicalx package, followed by the Geo2DR package.

```bash
cd modded_chemicalx
pip install -e .
cd ..
pip install git+https://github.com/paulmorio/geo2dr.git
```

## Code overview

- `main.py`: Script for running/evaluating DR-Augmented models. Automatically learns distributed representations if they are not available for given hyperparameters.
- `nondr_main.py`: Script for running/evaluating current state-of-the-art models without distributed representations
- `results_analysis`: Notebook containing summaries of the results files outputted by main.py and nondr_main.py. Contains plots, ablation study etc as reported in the paper.
- `train_dr_model.py`: Utilities for training models with distributed representations and cho
- `data`: Contains code related to inducing substructure patterns and learning distributed representations.
- `modded_chemicalx`: A modified version of the ChemicalX package, necessitated by updates to the TorchDrug API, and new dataloaders/models utilising the distributed representations of graphs. 
