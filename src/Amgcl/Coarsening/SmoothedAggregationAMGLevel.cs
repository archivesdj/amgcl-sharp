using Amgcl.Matrix;

namespace Amgcl.Coarsening;

public class SmoothedAggregationAMGLevel : IAMGLevel
{
    private readonly int _minGridSize;

    public SparseMatrixCSR A { get; private set; }
    public SparseMatrixCSR? R { get; private set; }
    public SparseMatrixCSR? P { get; private set; }
    public double[] Residual { get; private set; }
    public double[] Correction { get; private set; }

    public SmoothedAggregationAMGLevel(SparseMatrixCSR matrix, int minGridSize)
    {
        A = matrix ?? throw new ArgumentNullException(nameof(matrix));
        Residual = new double[matrix.Rows];
        Correction = new double[matrix.Rows];
        _minGridSize = minGridSize;
    }

    public IAMGLevel? Coarsen()
    {
        int coarseSize = ComputeCoarseSize();
        if (coarseSize < _minGridSize) return null; // Check minimum grid size

        SparseMatrixCSR P_0 = CreateTentativeProlongation(coarseSize);
        P = SmoothProlongation(P_0);
        R = CreateRestrictionOperator(P);
        SparseMatrixCSR A_c = ComputeCoarseMatrix();
        return CreateCoarseLevel(A_c);
    }

    private int ComputeCoarseSize()
    {
        return (A.Rows + 1) / 2; // 2:1 aggregation
    }

    private SparseMatrixCSR CreateTentativeProlongation(int coarseSize)
    {
        int fineSize = A.Rows;
        var p0Values = new List<double>();
        var p0ColIndices = new List<int>();
        var p0RowPointers = new List<int> { 0 };

        for (int i = 0; i < fineSize; i++)
        {
            int aggregateIdx = i / 2;
            p0Values.Add(1.0);
            p0ColIndices.Add(aggregateIdx);
            p0RowPointers.Add(p0Values.Count);
        }

        return new SparseMatrixCSR(fineSize, coarseSize, p0Values.ToArray(), p0ColIndices.ToArray(), p0RowPointers.ToArray());
    }

    private SparseMatrixCSR SmoothProlongation(SparseMatrixCSR P_0)
    {
        double omega = 0.7; // Damping factor
        SparseMatrixCSR D_inv = A.ComputeDiagonalInverse();
        SparseMatrixCSR D_inv_A = D_inv.MultiplyMatrix(A);
        SparseMatrixCSR I = A.CreateIdentityMatrix();
        SparseMatrixCSR temp = I.Subtract(D_inv_A.Scale(omega));
        return temp.MultiplyMatrix(P_0);
    }

    private SparseMatrixCSR CreateRestrictionOperator(SparseMatrixCSR prolongation)
    {
        return prolongation.Transpose();
    }

    private SparseMatrixCSR ComputeCoarseMatrix()
    {
        SparseMatrixCSR temp = A.MultiplyMatrix(P!);
        return R!.MultiplyMatrix(temp);
    }

    private IAMGLevel CreateCoarseLevel(SparseMatrixCSR A_c)
    {
        SmoothedAggregationAMGLevel coarseLevel = new SmoothedAggregationAMGLevel(A_c, _minGridSize);
        return coarseLevel;
    }
}