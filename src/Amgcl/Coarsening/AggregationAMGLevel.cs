using Amgcl.Matrix;

namespace Amgcl.Coarsening;

public class AggregationAMGLevel : IAMGLevel
{
     private readonly int _minGridSize;

    public SparseMatrixCSR A { get; private set; }
    public SparseMatrixCSR? R { get; private set; }
    public SparseMatrixCSR? P { get; private set; }
    public double[] Residual { get; private set; }
    public double[] Correction { get; private set; }

    public AggregationAMGLevel(SparseMatrixCSR matrix, int minGridSize)
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

        P = CreateProlongationOperator(coarseSize);
        R = CreateRestrictionOperator(P);
        SparseMatrixCSR A_c = ComputeCoarseMatrix();
        return CreateCoarseLevel(A_c);
    }

    private int ComputeCoarseSize()
    {
        return (A.Rows + 1) / 2; // 2:1 aggregation
    }

    private SparseMatrixCSR CreateProlongationOperator(int coarseSize)
    {
        int fineSize = A.Rows;
        var pValues = new List<double>();
        var pColIndices = new List<int>();
        var pRowPointers = new List<int> { 0 };

        for (int i = 0; i < fineSize; i++)
        {
            int aggregateIdx = i / 2;
            pValues.Add(1.0);
            pColIndices.Add(aggregateIdx);
            pRowPointers.Add(pValues.Count);
        }

        return new SparseMatrixCSR(fineSize, coarseSize, pValues.ToArray(), pColIndices.ToArray(), pRowPointers.ToArray());
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
        AggregationAMGLevel coarseLevel = new AggregationAMGLevel(A_c, _minGridSize);
        return coarseLevel;
    }
}