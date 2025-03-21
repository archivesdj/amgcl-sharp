using Amgcl.Matrix;

namespace Amgcl.Coarsening;

public class RugeStubenAMGLevel : IAMGLevel
{
    private readonly int _minGridSize;

    public SparseMatrixCSR A { get; private set; }
    public SparseMatrixCSR? R { get; private set; }
    public SparseMatrixCSR? P { get; private set; }
    public double[] Residual { get; private set; }
    public double[] Correction { get; private set; }

    public RugeStubenAMGLevel(SparseMatrixCSR matrix, int minGridSize)
    {
        A = matrix;
        Residual = new double[matrix.Rows];
        Correction = new double[matrix.Rows];
        _minGridSize = minGridSize;
    }

    public IAMGLevel? Coarsen()
    {
        var (coarseIndices, coarseSize) = SplitCoarseFinePoints();
        if (coarseSize < _minGridSize || coarseSize == 0 || coarseSize == A.Rows) return null; // Check minimum grid size

        P = CreateProlongationOperator(coarseIndices, coarseSize);
        R = CreateRestrictionOperator(P);
        SparseMatrixCSR A_c = ComputeCoarseMatrix();
        return CreateCoarseLevel(A_c);
    }

    private (List<int> coarseIndices, int coarseSize) SplitCoarseFinePoints()
    {
        int n = A.Rows;
        bool[] isCoarse = new bool[n];
        List<int> coarseIndices = new List<int>();
        HashSet<int> fineIndices = new HashSet<int>(Enumerable.Range(0, n));

        for (int i = 0; i < n; i++)
        {
            if (!fineIndices.Contains(i)) continue;

            double maxOffDiag = 0;
            for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
                if (A.ColIndices[j] != i)
                    maxOffDiag = Math.Max(maxOffDiag, Math.Abs(A.Values[j]));
            double threshold = 0.25 * maxOffDiag;

            isCoarse[i] = true;
            coarseIndices.Add(i);
            fineIndices.Remove(i);

            for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
            {
                int col = A.ColIndices[j];
                if (col != i && Math.Abs(A.Values[j]) > threshold && fineIndices.Contains(col))
                    fineIndices.Remove(col);
            }
        }

        return (coarseIndices, coarseIndices.Count);
    }

    private SparseMatrixCSR CreateProlongationOperator(List<int> coarseIndices, int coarseSize)
    {
        int n = A.Rows;
        var pValues = new List<double>();
        var pColIndices = new List<int>();
        var pRowPointers = new List<int> { 0 };
        Dictionary<int, int> coarseMap = coarseIndices.Select((c, idx) => (c, idx)).ToDictionary(x => x.c, x => x.idx);

        for (int i = 0; i < n; i++)
        {
            if (coarseIndices.Contains(i))
            {
                pValues.Add(1.0);
                pColIndices.Add(coarseMap[i]);
            }
            else
            {
                double sumWeights = 0;
                Dictionary<int, double> weights = new Dictionary<int, double>();
                for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
                {
                    int col = A.ColIndices[j];
                    if (coarseIndices.Contains(col) && col != i)
                    {
                        double weight = -A.Values[j];
                        weights[coarseMap[col]] = weight;
                        sumWeights += weight;
                    }
                }
                double diag = 0;
                for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
                    if (A.ColIndices[j] == i) diag = A.Values[j];

                if (sumWeights > 0)
                {
                    foreach (var kvp in weights)
                    {
                        pValues.Add(kvp.Value / sumWeights * diag);
                        pColIndices.Add(kvp.Key);
                    }
                }
            }
            pRowPointers.Add(pValues.Count);
        }

        return new SparseMatrixCSR(n, coarseSize, pValues.ToArray(), pColIndices.ToArray(), pRowPointers.ToArray());
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
        RugeStubenAMGLevel coarseLevel = new RugeStubenAMGLevel(A_c, _minGridSize);
        return coarseLevel;
    }
}