namespace Amgcl.Enums;

public enum SolverType
{
    // CPU solvers
    DampedJacobi,
    CG,
    BiCGSTAB,
    LUDirect,
    GaussSeidel,
    AMG,
    // GPU solvers
    DampedJacobiGPU,
    CGGPU,
    BiCGSTABGPU,
    LUDirectGPU,
    GaussSeidelGPU,
    AMGGPU,
}