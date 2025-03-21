using Amgcl.Matrix;
using Amgcl.Enums;
using Amgcl.Solver;

namespace Amgcl.Helpers;

public static class SolverFactory
{
    // Create a solver instance based on the specified SolverType
    public static ISolver CreateSolver(SolverType type, SparseMatrixCSR matrix)
    {
        switch (type)
        {
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
            default:
                throw new ArgumentException("Unknown solver type specified.");
        }
    }
}