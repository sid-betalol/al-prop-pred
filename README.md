# Molecular Property Prediction with Active Learning

This project implements various machine learning approaches for predicting molecular properties, including plain training, active learning, and multi-fidelity active learning. The model is based on the PaiNN (Polarizable Atom Interaction Neural Network) architecture and is trained on the QM9 dataset.

This project was developed and tested on:
- Apple M2 CPU
- Python 3.11

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sid-betalol/enthalpic-ai-challenge.git
   cd enthalpic-ai-challenge
   ```
2. Create a virtual environment and activate it
  ```bash
  conda create -n entalpic python=3.11
  conda activate entalpic
  ```
3. Install requirements
  ```bash
  pip install -r requirements.txt
  ```
If you are struggling with installing requirements on a Mac system, this [tutorial](https://youtu.be/UuMjJVqCMQo?si=HE7HxDa2pdSKjCRX) can be super useful.

## Usage
Make sure that you are in the `src` folder. 
```bash
cd src
```
Take a look at the input arguments and their default values in `src/train.py`

### Plain Training with Actual Labels
```bash
python train.py --epochs 10 --batch_size 32 --lr 0.0005
```
### Plain Training with Pre-trained Labels
```bash
python train.py --epochs 10 --batch_size 32 --lr 0.0005 --use_pretrained_labels
```
### Active Learning Training with Actual Labels
```bash
python train.py --use_active_learning --al_iterations 5 --points_per_iter 100 --acquisition uncertainty --use_al_true_labels
```
### Active Learning Training with Pretrained Labels
```bash
python train.py --use_active_learning --al_iterations 5 --points_per_iter 100 --acquisition uncertainty
```
### Multi-fidelity Active Learning
```bash
python train.py --use_active_learning --use_multi_fidelity_al --al_iterations 5 --points_per_iter 100
```

## Model Architecture
The surrogate model used in this project is based on the PaiNN (Polarizable Atom Interaction Neural Network) architecture. It consists of:

* An embedding layer for atomic numbers
* Multiple message-passing layers
* Update layers with gated equivariant blocks
* A final output layer for property prediction

The model processes both scalar and vector features, making it suitable for predicting molecular properties that depend on 3D structure.

## Pretrained Labellers
Two pretrained models are used as labellers: DimeNet++ and DimeNet. Both models are pretrained on the QM9 dataset. The initial intention was to use SchNet as the lower fidelity model, however there were package dependency issues with schnetpack.

## Acquisition Functions
The project implements several acquisition functions for active learning:

1. Uncertainty: Based on Monte Carlo dropout to estimate prediction uncertainty.
2. Expected Improvement (EI): Selects points that are expected to improve upon the current best observation.
3. BADGE (Batch Active learning by Diverse Gradient Embeddings): Combines uncertainty and diversity for batch selection.
4. Random: Randomly selects points (used as a baseline).
5. Multi-fidelity: Select points based on the disagreement between high and low-fidelity models.

To use a specific acquisition function(1-4), use the `--acquisition` flag followed by the function name (e.g., `--acquisition uncertainty`). The mutli-fidelity approach can be used by specifying the `--use_multi_fidelity_al` argument.

## Additional Features
* Resume training from a checkpoint using the `--resume` flag followed by the path to the checkpoint.
* Adjust learning rate and weight decay using `--lr` and `--weight_decay` flags.
* Set a random seed for reproducibility with the `--seed` flag.

For a full list of available options, run:
```bash
python train.py --help
```

## References
1. Duval, Alexandre, et al. "A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems." arXiv preprint arXiv:2312.07511 (2023).
2. Schütt, Kristof, Oliver Unke, and Michael Gastegger. "Equivariant message passing for the prediction of tensorial properties and molecular spectra." International Conference on Machine Learning. PMLR, 2021.
3. Gasteiger, Johannes, Janek Groß, and Stephan Günnemann. "Directional message passing for molecular graphs." arXiv preprint arXiv:2003.03123 (2020).
4. Gasteiger, Johannes, Chandan Yeshwanth, and Stephan Günnemann. "Directional message passing on molecular graphs via synthetic coordinates." Advances in Neural Information Processing Systems 34 (2021): 15421-15433.
5. Jing, Bowen, et al. "Learning from protein structure with geometric vector perceptrons." International Conference on Learning Representations. 2020.
6. Jing, Bowen, et al. "Equivariant graph neural networks for 3d macromolecular structure." arXiv preprint arXiv:2106.03843 (2021).
7. [PyTorch DimeNet++ Documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DimeNetPlusPlus.html#torch_geometric.nn.models.DimeNetPlusPlus)
8. https://github.com/lucidrains/alphafold3-pytorch
9. https://github.com/learningmatter-mit/NeuralForceField
10. https://github.com/atomistic-machine-learning/schnetpack
11. https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py
12. https://alectio.com/2022/12/20/active-learning-101-tricks-for-tuning-your-active-learning
