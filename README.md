# ProdNet: Fast Discovery of Matrix Multiplication Algorithms Using Lightweight Neural Networks

📄 Published Research Paper

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

This repository contains the official implementation of our published work: "ProdNet: Fast Discovery of Matrix Multiplication Algorithms Using Lightweight Neural Networks" (*this work has been published in 03 June 2026*)

link: https://link.springer.com/chapter/10.1007/978-3-032-21585-7_7

## Overview

ProdNet is a lightweight neural network framework for the automatic discovery of matrix multiplication algorithms.

Matrix multiplication is one of the most fundamental operations in scientific computing, machine learning, numerical simulation, computer graphics, and artificial intelligence. Despite decades of research, discovering efficient matrix multiplication algorithms remains one of the most challenging problems in computational mathematics.

ProdNet approaches this challenge from a different perspective: instead of relying on reinforcement learning, large search trees, or massive computational resources, it uses a compact supervised-learning framework that directly learns tensor decompositions corresponding to matrix multiplication algorithms.

The project demonstrates that efficient multiplication schemes can be discovered using a small neural network trained on synthetically generated data, making matrix multiplication algorithm discovery accessible to researchers without specialized hardware.



# The Problem

Given two matrices:

[
C = A \times B
]

the standard algorithm requires:

[
n^3
]

scalar multiplications for multiplying two (n \times n) matrices.

For example:

| Matrix Size | Standard Multiplications |
| ----------- | ------------------------ |
| 2×2         | 8                        |
| 3×3         | 27                       |

Reducing the number of scalar multiplications has been a long-standing research objective because matrix multiplication is a core operation behind:

* Deep learning training
* Large Language Models (LLMs)
* Scientific computing
* Optimization algorithms
* Numerical simulations
* Computer graphics

Even small reductions in multiplication count can translate into significant computational savings when scaled to large workloads.

Historically, discovering fast matrix multiplication algorithms required extensive mathematical analysis or computationally expensive search methods.

ProdNet investigates whether a lightweight neural network can automatically discover such algorithms through learning.


## Matrix Multiplication as Tensor Decomposition

Modern approaches to fast matrix multiplication reformulate matrix multiplication as a tensor decomposition problem. Instead of directly searching for multiplication algorithms, the matrix multiplication tensor can be decomposed into a sum of rank-1 tensors:

<img width="400" height="50" alt="image" src="https://github.com/user-attachments/assets/c9e95ab1-4ccd-4295-875a-9e044bb3c1a3" />


where (R) corresponds to the number of scalar multiplications required by the algorithm.

The multiplication of 2 × 2 matrices can be performed with 7 multiplication instead of 8 by following Strassen’s algorithm.

<img width="550" height="100" alt="image" src="https://github.com/user-attachments/assets/48383284-22f3-45f1-ae60-2dc402e1a795" />

<img width="300" height="100" alt="image" src="https://github.com/user-attachments/assets/25e507c1-df07-4f8a-b142-ddf96d08968e" /> <img width="300" height="100" alt="image" src="https://github.com/user-attachments/assets/a1a9e102-6f61-44d6-8674-d764650db17d" /> <img width="300" height="100" alt="image" src="https://github.com/user-attachments/assets/11be3047-333e-4437-8f8f-3cdd4d79d45c" />

This decomposition can be translated to a multiplication algorithm as follows, where the matrices’ values of U, V and W are special for a, b, and c respectively.

<img width="427" height="486" alt="image" src="https://github.com/user-attachments/assets/d99c9a6a-0ed2-4b26-ad8d-1697b3d1016f" />


# Methodology

ProdNet reformulates matrix multiplication discovery as a supervised learning problem.

The network architecture consists of:

1. A linear transformation of matrix A
2. A linear transformation of matrix B
3. A multiplication layer
4. A reconstruction layer producing the output matrix

   <img width="596" height="420" alt="image" src="https://github.com/user-attachments/assets/a6e0b846-3920-46f0-9c3b-a86d9d3ab7d3" />


The hidden neurons represent intermediate multiplication terms:

[ m_i = (U_i^T A)(V_i^T B) ]

while the output layer combines these terms using matrix W to reconstruct the final product matrix.

The learned matrices: [ (U,V,W) ]

correspond directly to tensor decomposition factors and can be translated into symbolic matrix multiplication algorithms.


## Discrete Algorithm Discovery

A major challenge is ensuring that learned coefficients correspond to valid algorithmic structures.

To address this, ProdNet introduces a custom regularization objective that encourages weights to converge toward:

[
{-1,0,1}
]

rather than arbitrary continuous values.

This allows the trained network to discover interpretable multiplication algorithms whose weights can be converted directly into mathematical expressions.

The final optimization objective combines:

* Mean Squared Error (MSE)
* Discrete-value regularization

resulting in accurate matrix multiplication while simultaneously discovering sparse algorithmic structures.



# Results

## 2×2 Matrix Multiplication

ProdNet successfully discovers multiplication algorithms requiring **7 Multiplications** instead of the classical 8 Multiplications

The discovered solutions are equivalent in multiplication count to Strassen-type decompositions and demonstrate that the model can automatically recover low-rank matrix multiplication structures.

- Algorithm 1 was discovered in 500 iterations:

<img width="519" height="398" alt="image" src="https://github.com/user-attachments/assets/69b5d707-d73f-4a4a-8324-1013968e370b" />

- Algorithm 2 was discovered in 2500 iterations:

<img width="519" height="391" alt="image" src="https://github.com/user-attachments/assets/76d6dd09-68fb-4a05-baa0-9903a04aa57f" />

- Model Convergence During Training Using MSE Without Integer Rounding of Weights.

<img width="626" height="379" alt="image" src="https://github.com/user-attachments/assets/7eb8875a-89ba-4559-8d82-dffd75eab977" />

- Model Convergence During Training Using MSE With Integer Rounding of Weights:

<img width="626" height="379" alt="image" src="https://github.com/user-attachments/assets/9b5fb616-6925-45a7-ba71-cf712e5bb246" />

- Model convergence using different weight initialization values and learning rate:

<img width="626" height="379" alt="image" src="https://github.com/user-attachments/assets/bea3545a-0a7d-41ec-9902-34b39931e9ba" />


## 3×3 Matrix Multiplication

ProdNet successfully discovers multiplication algorithms requiring **23 Multiplications** instead of the classical 27 Multiplications.

The algorithm was discovered automatically through training without embedding prior knowledge of existing matrix multiplication schemes.

- The algorithm discovered for 3 by 3 matrices size:

<img width="508" height="823" alt="image" src="https://github.com/user-attachments/assets/1a33944c-a753-4ab4-b47e-1a6d893dfcb7" />

- Model convergence for 3 by 3 matrix size:

<img width="626" height="379" alt="image" src="https://github.com/user-attachments/assets/f60866ef-f703-4345-96b8-d023bc4a33d7" />

## Computational Efficiency

Experiments were performed using modest computational resources, including:

* NVIDIA Tesla T4 GPUs
* Google Colab
* Kaggle Notebooks

Unlike large-scale search systems, ProdNet does not require:

* Transformer architectures
* Monte Carlo Tree Search (MCTS)
* Reinforcement learning pipelines
* Specialized TPU clusters

This significantly lowers the barrier to experimentation and reproduction.



# Impact

Fast matrix multiplication algorithms have influenced computational mathematics for more than five decades.

The significance of this research extends beyond matrix multiplication itself.

Potential applications include:

### Efficient Deep Learning

Matrix multiplication dominates the computational cost of training modern neural networks, including Transformers and Large Language Models.

### Reduced Computational Cost

Discovering lower-rank multiplication schemes can reduce the number of arithmetic operations required by many machine learning workloads.

### Accessible Algorithm Discovery

ProdNet provides a lightweight alternative to computationally intensive algorithm-discovery systems, enabling broader participation in this area of research.

### Neural-Guided Mathematical Discovery

The project demonstrates how neural networks can be used not only for prediction tasks but also for discovering mathematical structures and algorithms.



# Key Features

* Lightweight neural network architecture
* Supervised-learning-based algorithm discovery
* Direct correspondence between network weights and multiplication algorithms
* Discrete weight regularization
* Automatic discovery of low-rank matrix multiplication schemes
* Reproducible experiments on commodity hardware
* Tensor decomposition interpretation of learned solutions



# Future Directions

Future work includes:

* Discovery of 4×4 and larger multiplication algorithms
* Sparse tensor decomposition strategies
* Discrete optimization techniques
* Exhaustive search in reduced parameter spaces
* Hardware-aware multiplication algorithm discovery
* Exploration of alternative regularization objectives



# Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{ouissam2025prodnet,
  title={ProdNet: A Lightweight Network for Fast Discovery of Matrix Multiplication Algorithms},
  author={Ouissam Lakas, Badia and Berdjouh, Chemousse and Bounane, Khadra and Lamine Kherfi, Mohammed and Aiadi, Oussama and Brahim Belhouari, Samir},
  booktitle={International Conference on Intelligent Systems and Pattern Recognition},
  pages={95--108},
  year={2025},
  organization={Springer}
}
```



## Conclusion

ProdNet demonstrates that efficient matrix multiplication algorithms can be discovered using a compact neural network trained with supervised learning and discrete regularization. By rediscovering low-rank algorithms for both 2×2 and 3×3 matrix multiplication while requiring only modest computational resources, ProdNet provides an accessible framework for neural-guided algorithm discovery and computational mathematics research.
