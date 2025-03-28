using Amgcl.Matrix;

namespace Amgcl.Coarsening;

public class VanillaAMGLevel : IAMGLevel
{
    private readonly int _minGridSize;
    public SparseMatrixCSR A { get; private set; }
    public SparseMatrixCSR? R { get; private set; }
    public SparseMatrixCSR? P { get; private set; }
    public double[] Residual { get; private set; }
    public double[] Correction { get; private set; }

    public VanillaAMGLevel(SparseMatrixCSR matrix, int minGridSize)
    {
        A = matrix;
        Residual = new double[matrix.Rows];
        Correction = new double[matrix.Rows];
        _minGridSize = minGridSize;
    }

    public IAMGLevel? Coarsen()
    {
        int coarseSize = ComputeCoarseSize();
        if (coarseSize < _minGridSize) return null; // Check minimum grid size

        R = CreateRestrictionOperator(coarseSize);
        P = CreateProlongationOperator(R);
        SparseMatrixCSR A_c = ComputeCoarseMatrix();
        return CreateCoarseLevel(A_c);
    }

    private int ComputeCoarseSize()
    {
        return A.Rows / 2; // 2:1 coarsening
    }

    private SparseMatrixCSR CreateRestrictionOperator(int coarseSize)
    {
        int fineSize = A.Rows;
        double[] rValues = new double[fineSize];
        int[] rColIndices = new int[fineSize];
        int[] rRowPointers = new int[coarseSize + 1];
        for (int i = 0; i < coarseSize; i++)
        {
            rRowPointers[i] = 2 * i;
            rValues[2 * i] = 0.5;
            rValues[2 * i + 1] = 0.5;
            rColIndices[2 * i] = 2 * i;
            rColIndices[2 * i + 1] = 2 * i + 1;
        }
        rRowPointers[coarseSize] = fineSize;
        return new SparseMatrixCSR(coarseSize, fineSize, rValues, rColIndices, rRowPointers);
    }

    private SparseMatrixCSR CreateProlongationOperator(SparseMatrixCSR restriction)
    {
        return restriction.Transpose();
    }

    private SparseMatrixCSR ComputeCoarseMatrix()
    {
        SparseMatrixCSR temp = A.MultiplyMatrix(P!);
        return R!.MultiplyMatrix(temp);
    }

    private IAMGLevel CreateCoarseLevel(SparseMatrixCSR A_c)
    {
        VanillaAMGLevel coarseLevel = new VanillaAMGLevel(A_c, _minGridSize);
        return coarseLevel;
    }
}