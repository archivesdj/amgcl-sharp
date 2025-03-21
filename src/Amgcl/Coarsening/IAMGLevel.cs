using Amgcl.Matrix;

namespace Amgcl.Coarsening;

public interface IAMGLevel
{
    SparseMatrixCSR A { get; }        // Matrix at this level
    SparseMatrixCSR? R { get; }        // Restriction operator
    SparseMatrixCSR? P { get; }        // Prolongation operator
    double[] Residual { get; }        // Residual vector
    double[] Correction { get; }      // Correction vector

    // Create a coarser level
    IAMGLevel? Coarsen();
}