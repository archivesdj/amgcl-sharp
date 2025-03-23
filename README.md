# amgcl-sharp

## What is AMGCL?

[AMGCL](https://github.com/ddemidov/amgcl) is a C++ header-only library designed to efficiently solve large sparse linear systems. It implements the Algebraic Multigrid (AMG) method, which is particularly well-suited for solving systems of equations arising from the discretization of partial differential equations (PDEs) on unstructured grids. AMG can be used as a “black-box solver” without geometric information, making it valuable for various computational problems. It is often used not as a standalone solver but as a preconditioner for iterative solvers such as Conjugate Gradients, BiCGStab, and GMRES.

### Key Features

- Header-only design: Minimal external dependencies, making installation straightforward. Apart from the Boost library, AMGCL has very few additional dependencies.
- Support for multiple backends: AMGCL constructs the AMG hierarchy on the CPU and then transfers it to different backends for accelerating the solution phase. Supported backends include:
  - OpenMP: Multi-threaded CPU parallel processing.
  - OpenCL: Utilizes GPUs and other parallel processors.
  - CUDA: High-performance computation on NVIDIA GPUs.
  - Users can also define custom backends to integrate tightly with their code.
- Shared and distributed memory: Works in both shared-memory (single-node) and distributed-memory (multi-node via MPI) environments.
- Scalability: A compile-time policy-based design allows users to customize AMG components or add new features. A runtime interface is also provided for added flexibility.
- Performance: AMGCL delivers competitive performance compared to other AMG implementations.

### Example Usage

A simple example of solving the Poisson equation using a BiCGStab solver with smoothed aggregation AMG as a preconditioner:

```cpp
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

typedef amgcl::backend::builtin<double> Backend;
typedef amgcl::make_solver<
    amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
    amgcl::solver::bicgstab<Backend>
> Solver;
```

Since AMGCL is a C++ header-only library, using it in C# requires calling native C++ code. A simple implementation for this can be found in [Amgcl.Net](https://github.com/archivesdj/Amgcl.Net).

## Implementing Similar Functionality in C#

This project was initiated to explore implementing similar functionality in C#.

Directly implementing an Algebraic Multigrid (AMG) algorithm in C# is quite complex, but it is possible when approached step by step. Here, we attempt to implement a simplified AMG solver in C#, incorporating the following key elements:

- Sparse Matrix Representation: Using Compressed Sparse Row (CSR) format (`SparseMatrixCSR`)
- AMG Components:
  - Coarsening (`IAMGLevel`)
  - Relaxation methods (`ISolver`)
	•	Construction of the multigrid hierarchy and execution of V-cycle (`AMGSolver`)
	•	Parallel processing using ILGPU for performance optimization (`AMGSolverGPU`)

This approach aims to provide a foundational AMG implementation in C#, making it easier to experiment with and adapt for various computational problems.