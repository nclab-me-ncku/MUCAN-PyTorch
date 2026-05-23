# **M**asked __**U**__-Net-Based **C**ycle-Consistent **A**dversarial **N**etworks (MUCAN) - PyTorch
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  

Feel free to utilize the code and cite this paper.
Jhih-Siang Huang, Chih-Hsuan Wu, Wei-Chen Chen, Yueh-Huan Lee, and Shih-Hung Yang, "Robust Motor Decoding Using Distribution Alignment Without Recalibrating Intracortical Brain-Computer Interfaces," _IEEE Transactions on Cognitive and Developmental Systems_, Accepted, 2026.


## ABSTRACT
Intracortical brain-computer interfaces (iBCIs) allow paralyzed individuals to regain motor functions by translating neural activity into control commands for assistive devices. However, neural variability caused by biological factors and electrode connectivity issues disrupts the mapping between neural signals and motor outputs, reducing decoding performance. While supervised recalibration of the neural decoder on subsequent days can mitigate this variability, the process often requires specialized personnel for operating recording systems and acquiring labeled data, which poses practical challenges. To address this issue, we propose Masked U-net-based Cycle-consistent Adversarial Networks (MUCAN), a framework designed to maintain the neural-to-kinematic mapping by aligning daily neural activity distributions with a reference day, thereby eliminating the need for recalibration. MUCAN employs a masking strategy to extract features that remain consistent across days, effectively mitigating daily variability in neural recording conditions. Additionally, it leverages the U-net architecture to extract hierarchical features and preserve fine-grained details of the neural activity, ensuring accurate distribution alignment. Experimental results demonstrate that MUCAN successfully maintains the neural-to-kinematic mapping and outperforms state-of-the-art iBCIs, achieving an average performance improvement of 15.69% across eight publicly available nonhuman primate datasets. MUCAN exhibits robustness against neural signal loss, even with 50 unexpectedly disabled recording channels. By projecting firing rates onto the first two principal components for visualizing latent trajectories, MUCAN effectively preserves the underlying structure of neural population activity over time. This study provides a robust approach to sustaining motor decoding performance without the necessity for continuous recalibration, enhancing the practicality of iBCIs for real-world applications.

## How to use
### Install dependencies

Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```
### Read the tutorial

Please read `tutorial.ipynb` first. It provides a step-by-step guide to help you understand how to use this project.
