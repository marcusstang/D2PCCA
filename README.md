# Deep Dynamic Probabilistic Canonical Correlation Analysis (D2PCCA)

This repository contains the implementation of **"Deep Dynamic Probabilistic Canonical Correlation Analysis"** by Shiqin Tang, Shujian Yu, Yining Dong, and S. Joe Qin. The paper introduces a model that integrates deep learning with probabilistic modeling to analyze nonlinear dynamical systems.

![graphical_model](Images/main_fig.png)

## Abstract
This paper presents Deep Dynamic Probabilistic Canonical Correlation Analysis (D$^2$PCCA), a model that integrates deep learning with probabilistic modeling to analyze nonlinear dynamical systems. Building on the probabilistic extensions of Canonical Correlation Analysis (CCA), D$^2$PCCA captures nonlinear latent dynamics and supports enhancements such as KL annealing for improved convergence and normalizing flows for a more flexible posterior approximation. D$^2$PCCA naturally extends to multiple observed variables, making it a versatile tool for encoding prior knowledge about sequential datasets and providing a probabilistic understanding of the system’s dynamics. Experimental validation on real financial datasets demonstrates the effectiveness of D$^2$PCCA and its extensions in capturing latent dynamics.

## Implemented Models
- (Multiset) Dynamic Probabilistic CCA (DPCCA) 
- (Multiset) Deep Dynamic Probabilistic CCA (D$^2$PCCA)

## Installation

**Clone the repository**:
```bash
git clone https://github.com/marcusstang/D2PCCA.git
cd D2PCCA
```

## Data Acquisition and Preprocessing

1. **Download the Dataset**:  
   Obtain the [S&P 500 Stock Data Dataset](https://www.kaggle.com/datasets/camnugent/sandp500) from Kaggle. Unzip the downloaded file and place the CSV files in the `data/` directory.

2. **Preprocess the Data**:  
   Execute the preprocessing script to clean and prepare the data:
   ```bash
   Rscript data/get_data.R
   ```
   This script will load the raw stock data, handle missing values, and output the processed data files for training the model.


## Acknowledgments

This research work is supported by a Math and Application Project (2021YFA1003504) under the National Key R&D Program, a Collaborative Research Fund by the RGC of Hong Kong (Project No. C1143-20G), a grant from the Natural Science Foundation of China (U20A20189), a grant from the ITF - Guangdong-Hong Kong Technology Cooperation Funding Scheme (Project Ref. No. GHP/145/20), and an InnoHK initiative of The Government of the HKSAR for the Laboratory for AI-Powered Financial Technologies.

We would also like to thank the contributors and maintainers of PyTorch and Pyro for their excellent machine learning libraries.

Our implementation is inspired by the following works:

- [Pyro DMM Example](https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm.py): This code provides an excellent foundation for implementing deep state space models using Pyro. We adapted some of their model structures and training routines to suit our framework. We would like to acknowledge these contributions and thank the authors for making their code publicly available.



