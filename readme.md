# Damage Detection Using Autoencoder (AE) and Generative Adversarial Network (GAN) for nonlinear systems

This repository contains the code implementation for **Deep learning architectures for data-driven damage
detection in nonlinear dynamic systems under random vibrations** based on the methods described in the journal paper.


## Introduction
Structural damage detection is a critical task in engineering. This project uses deep learning techniques (AE and GAN) to detect damage in nonlinear systems  by analyzing time-series data based on 1D CNN. These NNs are trained in an unsupervised manner and their performances are compared for different nonlinear systems. The time-series data for the nonlinear system are generated using numerical computations. The following nonlinear systems are considered 
- 1-DOF Duffing system
- 2-DOF Duffing system
- Vibration isolator with super elastic hysteresis and negative stiffness
The damage time-series dataset for these systems were obtained by systematically reducing the stiffness of the system. Finally the AE, GAN architectures were tested on a experimental dataset of a Magneto-elastic clamped beam system.

Key contributions of the paper:
- AE for feature extraction and damage detection.
- Discriminator- part of GAN for detecting damaged time series signals.

## Included in this repo
- 1D Autoencoder and 1D GAN for time-series data for a 2-DOF Duffing oscillator.
- Includes the [data](https://drive.google.com/file/d/1BZEsVKchV6-oSNpYNMTaZ1Tm0a-lnWl3/view?usp=share_link) for the 2DOF Duffing oscillator.
- Provides visualization of results (e.g., loss curves, reconstructed signals).


## Citation
If you use this code, please cite:
```bash
@article{Joseph_Quaranta_Carboni_Lacarbonara_2024, title={Deep learning architectures for data-driven damage detection in nonlinear dynamic systems under random vibrations}, volume={112}, ISSN={1573-269X}, DOI={10.1007/s11071-024-10270-1}, number={23}, journal={Nonlinear Dynamics}, author={Joseph, Harrish and Quaranta, Giuseppe and Carboni, Biagio and Lacarbonara, Walter}, year={2024}, month=dec, pages={20611–20636} }