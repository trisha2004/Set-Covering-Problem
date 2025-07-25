## 
# Genetic Algorithm for the Set Covering Problem (SCP)

## 🧠 Introduction

The Set Covering Problem (SCP) is a classical NP-hard optimization problem. The goal is to select the minimum number of subsets from a given collection such that their union covers a universe set completely.

This project implements a **Genetic Algorithm (GA)** to approximate solutions for SCP under a **time constraint of 45 seconds**, incorporating multiple enhancements to improve convergence speed and solution quality.

---

## 📌 Problem Definition

- **Universe Set (U):** Integers from 1 to 100.
- **Subset Collection (S):** A collection of 50 to 350 subsets randomly generated.
- **Objective:** Select the minimum number of subsets from S whose union equals U.

---

## 🧬 Genetic Algorithm Components

### Solution Representation
- Binary vector of size |S|.
- 1 indicates inclusion of subset; 0 means exclusion.

### Fitness Function
```math
Fitness = |Covered Elements| - 5 * |Uncovered Elements| - 0.5 * |Subsets Used|
```
- Rewards coverage
- Penalizes uncovered elements and large subset count

### Initialization
- Population of 50 individuals, randomly generated

### Selection
- **Tournament Selection**: Choose the best among 3 randomly selected individuals

### Crossover
- **Single-point crossover** to combine parent solutions

### Mutation
- **Bit-flip mutation** with small probability for genetic diversity

---

## 🧪 Experimental Setup

- **Population Sizes Tested:** 30, 50, 100
- **Generation Counts Tested:** 20, 50, 100
- **Time Limit:** 45 seconds
- **Runs:** 10 per configuration

---

## ⚙️ Key Enhancements

1. **Early Termination**: Stops if fitness doesn't improve for 10 generations
2. **Time Limit Check**: Stops if 45s is exceeded
3. **Parameter Tuning**: Mutation rate & population size adjusted for best trade-offs
4. **Multiple Configurations**: Ran experiments for different S sizes

---

## 📈 Results Summary

- Best results achieved with **population size = 50** and **50 generations**
- Early stopping significantly reduced run-time without sacrificing solution quality
- Larger populations explored more, but increased computation cost
- Mutation rate tuning was critical to avoid local optima

---

## 🚀 How to Run

1. Clone the repository
2. Compile and run the main Python/C++/Java (specify language) script:
    ```bash
    python3 main.py
    ```
3. Set parameters in the config section if needed

---



## 🙌 Acknowledgments

Developed by Trisha Reddy and Amrita Pochiraju as part of academic coursework.

---


<
