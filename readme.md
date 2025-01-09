# Damage Detection Using Autoencoder and GAN

This repository contains the code implementation for **damage detection in nonlinear systems** using Autoencoder (AE) and Generative Adversarial Network (GAN), based on the methods described in my journal paper.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Citation](#citation)

## Introduction
Structural damage detection is a critical task in engineering. This project uses deep learning techniques (AE and GAN) to detect damage in nonlinear systems, such as the Duffing oscillator, by analyzing time-series data.

Key contributions:
- Autoencoder for feature extraction and anomaly detection.
- GAN for generating realistic damage scenarios for robust detection.

## Features
- Implements a 1D Autoencoder and 1D GAN for time-series data.
- Includes sample data for a Duffing oscillator.
- Provides visualization of results (e.g., loss curves, reconstructed signals).

## Installation
Clone this repository and install the required dependencies:
```bash
git clone https://github.com/your-username/DamageDetectionAEGAN.git
cd DamageDetectionAEGAN
pip install -r requirements.txt