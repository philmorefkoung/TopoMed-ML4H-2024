# TopoMed-ML4H-2024
 - This repository contains the code for __TopoMed__ one of two models in our paper "Enhancing Medical Imaging with Topological Methods in Limited-Data Settings"
 - Our project introduces the novel addition of topological data analysis (TDA) to imrpove the robustness of deep learning models for medical imaging classification 
 - __TopoMed__ is a Multilayer Perceptron (MLP) that is computationally __much__ faster with relatively the same, if not better, accuracy and AUROC than vanilla convolutional neural networks (CNN)
 - Accepted by Machine Learning for Health 2024

Data
---
Datasets, including the original experiments, can be found on the [MedMNIST website](https://medmnist.com/)

Usage
---
Please see [requirements.txt](requirements.txt) and [environment.yml](environment.yml) for package and dependency requirements

Results and Figures
---
Our full results and figures can be found in the paper, this repository contains the code for our standalone topological data model which achieves competitive results with CNNs and even outperforms them in certain cases

Methodology
---
We use Betti numbers, a subfield of topological data analysis, to generate a vector for each image that represents the topological evolution throughout a process called sublevel filtration. For each level of the filtration, we count the number of loops and connected components and use that as a direct embedding into our MLP or Augmented CNN. To add, since our inputs are a direct embedding, TopoMed is orders of magnitude faster than traditional CNNs and achieve comparable (within 5%) results while sometimes even outperforming them. A visual example of sublevel filtration can be found below:

![Path224](https://github.com/user-attachments/assets/3764bb59-0771-468d-9c67-1efa7caa52d6) <br />
*Threshold <= (color value of pixel activated)
