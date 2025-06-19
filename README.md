# PINN-WSGG Demo Repository

This repository contains a demonstration of the Physics-Informed Neural Network (PINN)-based Weighted Sum of Gray Gases (WSGG) model implementation.

Key features demonstrated:
- PINN-WSGG model implementation
- 1D Radiative Transfer Equation (RTE) solver
- Comparison with Line-by-Line (LBL) benchmarks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the demo script:
```bash
python pinn_wsgg_demo.py
```

This will:
1. Load the pre-trained PINN-WSGG model from `pinn_wsgg/data/wsgg.json`
2. Solve a 1D RTE problem
3. Compare results with LBL solution
4. Generate comparison plots

## Demo Output

![Demo Output](demo.png)

## Project Structure

```
pinn_wsgg/          # Core implementation
├── nn_wsgg.py      # Neural network WSGG model
├── mlp.py          # Multi-layer perceptron
├── rte.py          # Radiative transfer solver
├── plot_setup.py   # Plotting configuration
└── data/           # Model data
    └── wsgg.json   # Pre-trained model parameters

pinn_wsgg_demo.py   # Demonstration script
data/               # Output data
    ├── profiles.npy  # Solution profiles
    └── results.npy   # Comparison results
```

## Cite as

If you use PINN-WSGG in your research, please cite as follows,

```bibtex
@article{chenDevelopmentValidationPhysicsinformed2025,
  title = {Development and Validation of a Physics-Informed Neural Network-Based {{WSGG}} Model for Multi-Species Gas Mixtures},
  author = {Chen, Wei and Yang, Runze and Ren, Tao and Zhao, Changying},
  year = {2025},
  month = nov,
  journal = {International Journal of Heat and Mass Transfer},
  volume = {251},
  pages = {127328},
  issn = {0017-9310},
  doi = {10.1016/j.ijheatmasstransfer.2025.127328},
}
```

W. Chen, R. Yang, T. Ren, and C. Zhao, "Development and validation of a physics-informed neural network-based WSGG model for multi-species gas mixtures," *International Journal of Heat and Mass Transfer*, vol. 251, p. 127328, Nov. 2025, doi: 10.1016/j.ijheatmasstransfer.2025.127328.

## License

MIT License - See LICENSE file for details.
