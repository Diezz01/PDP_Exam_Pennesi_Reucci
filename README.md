# PDP_Exam_Pennesi_Reucci
This repository contains the code for the PDP exam of Pennesi and Reucci. <br>
The project is structured as follows:
- `Dataset_airline`: is the dataset retrived from Kaggle, containing the information about the flights of the airline.
  - `load_dataset_airline.cpp`: is the script used to load and preprocess the dataset using normalization and one-hot encoding for the categorical features. This create a binary file containing the preprocessed dataset, which is used for training the logistic regression model.
- `logistic_regression_cpu.cpp`: is the script used to implement the logistic regression algorithm on CPU, using gradient descent for optimization.
- `logistic_regression.cu`: is the script used to implement the logistic regression algorithm on GPU, using CUDA for parallelization and optimization using global memory.
- `logistic_regression_splitted.cu`: is the script used to implement the logistic regression algorithm on GPU, using CUDA for parallelization and optimization using dedicated kernel for computing linear combination, sigmoid function and gradients.
- `logistic_regression_shared.cu`: is the script used to implement the logistic regression algorithm on GPU, using CUDA for parallelization and optimization using shared memory.  
- `logistic_regression_minibatch.cu`: is the script used to implement the logistic regression algorithm on GPU, using CUDA for parallelization and optimization using mini-batch gradient descent and shared memory.

- To compile the cpp code, you can use the following command:<br>
``g++ -O3 -std=c++17 logistic_regression_cpu.cpp -o logistic_regression_cpu`` <br>
OR<br>
``cl /O2 /Ot /arch:AVX2 /EHsc logistic_regression_cpu.cpp``<br>
- To compile the CUDA code, you can use the following command:
``nvcc -O3 -use_fast_math -arch=sm_89 logistic_regression_shared.cu -o logistic_regression_shared``