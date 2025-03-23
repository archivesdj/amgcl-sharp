using Amgcl.Matrix;
using Amgcl.Enums;
using Amgcl.Solver;
using ILGPU;
using ILGPU.Runtime;

namespace Amgcl.Helpers;

public static class SolverFactory
{
    // Create a solver instance based on the specified SolverType
    public static ISolver CreateSolver(SolverType type, SparseMatrixCSR matrix, Accelerator? accelerator = null)
    {
        switch (type)
        {
            // CPU solvers
            case SolverType.DampedJacobi:
                return new DampedJacobiSolver(matrix);
            case SolverType.CG:
                return new ConjugateGradientSolver(matrix);
            case SolverType.BiCGSTAB:
                return new BiCGSTABSolver(matrix);
            case SolverType.LUDirect:
                return new LUDirectSolver(matrix);
            case SolverType.GaussSeidel:
                return new GaussSeidelSolver(matrix);
            case SolverType.AMG:
                return new AMGSolver(matrix);
            // GPU solvers
            case SolverType.DampedJacobiGPU:
                return new DampedJacobiSolverGPU(matrix, accelerator);
            case SolverType.CGGPU:
                return new ConjugateGradientSolverGPU(matrix, accelerator);
            case SolverType.BiCGSTABGPU:
                return new BiCGSTABSolverGPU(matrix, accelerator);
            case SolverType.LUDirectGPU:
                return new LUDirectSolverGPU(matrix, accelerator);
            case SolverType.GaussSeidelGPU:
                return new GaussSeidelSolverGPU(matrix, accelerator);
            case SolverType.AMGGPU:
                return new AMGSolverGPU(matrix, accelerator);
            default:
                throw new ArgumentException("Unknown solver type specified.");
        }
    }
}