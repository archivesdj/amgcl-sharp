using Amgcl.Matrix;

namespace Amgcl.Coarsening;

public class VanillaAMGLevel : IAMGLevel
{
    public SparseMatrixCSR A { get; private set; }
    public SparseMatrixCSR? R { get; private set; } = null;
    public SparseMatrixCSR? P { get; private set; } = null;
    public double[] Residual { get; private set; }
    public double[] Correction { get; private set; }

    // Constructor: Initialize level with matrix
    public VanillaAMGLevel(SparseMatrixCSR matrix)
    {
        A = matrix;
        Residual = new double[matrix.Rows];
        Correction = new double[matrix.Rows];
    }

    // Create coarse level (simple aggregation-based coarsening)
    public IAMGLevel? Coarsen()
    {
        int nFine = A.Rows;
        int nCoarse = nFine / 2; // Simple 2:1 coarsening for demonstration

        if (nCoarse < 3) return null;

        // Restriction (R): Average fine grid points to coarse grid
        double[] rValues = new double[nFine];
        int[] rColIndices = new int[nFine];
        int[] rRowPointers = new int[nCoarse + 1];
        for (int i = 0; i < nCoarse; i++)
        {
            rRowPointers[i] = 2 * i;
            rValues[2 * i] = 0.5;     // Aggregate two fine points
            rValues[2 * i + 1] = 0.5;
            rColIndices[2 * i] = 2 * i;
            rColIndices[2 * i + 1] = 2 * i + 1;
        }
        rRowPointers[nCoarse] = nFine;
        SparseMatrixCSR R = new SparseMatrixCSR(nCoarse, nFine, rValues, rColIndices, rRowPointers);

        // Prolongation (P): Transpose of R
        SparseMatrixCSR P = R.Transpose();

        // Coarse matrix A_coarse = R * A * P
        SparseMatrixCSR ACoarse = R.MultiplyMatrix(A.MultiplyMatrix(P)); // R * (A * P)

        // Create new level
        VanillaAMGLevel coarseLevel = new VanillaAMGLevel(ACoarse);
        coarseLevel.R = R;
        coarseLevel.P = P;
        return coarseLevel;
    }
}