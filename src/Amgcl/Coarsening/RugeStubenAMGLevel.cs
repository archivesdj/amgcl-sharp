using Amgcl.Matrix;

namespace Amgcl.Coarsening;

public class RugeStubenAMGLevel : IAMGLevel
{
    public SparseMatrixCSR A { get; private set; }
    public SparseMatrixCSR? R { get; private set; } = null;
    public SparseMatrixCSR? P { get; private set; } = null;
    public double[] Residual { get; private set; }
    public double[] Correction { get; private set; }

    // Constructor: Initialize level with matrix
    public RugeStubenAMGLevel(SparseMatrixCSR matrix)
    {
        A = matrix;
        Residual = new double[matrix.Rows];
        Correction = new double[matrix.Rows];
    }

    // Coarsen: Implement Ruge-Stuben coarsening with direct interpolation
    public IAMGLevel? Coarsen()
    {
        int n = A.Rows;

        // Step 1: Identify strong connections and split into C/F sets
        bool[] isCoarse = new bool[n];
        List<int> coarseIndices = new List<int>();
        HashSet<int> fineIndices = new HashSet<int>(Enumerable.Range(0, n));

        for (int i = 0; i < n; i++)
        {
            if (!fineIndices.Contains(i)) continue;

            double maxOffDiag = 0;
            for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
            {
                if (A.ColIndices[j] != i)
                    maxOffDiag = Math.Max(maxOffDiag, Math.Abs(A.Values[j]));
            }
            double threshold = 0.25 * maxOffDiag;

            isCoarse[i] = true;
            coarseIndices.Add(i);
            fineIndices.Remove(i);

            for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
            {
                int col = A.ColIndices[j];
                if (col != i && Math.Abs(A.Values[j]) > threshold && fineIndices.Contains(col))
                {
                    fineIndices.Remove(col);
                }
            }
        }

        int nCoarse = coarseIndices.Count;
        if (nCoarse == 0 || nCoarse == n) return null;

        // Step 2: Build prolongation operator P (direct interpolation)
        List<double> pValues = new List<double>();
        List<int> pColIndices = new List<int>();
        List<int> pRowPointers = new List<int> { 0 };
        Dictionary<int, int> coarseMap = coarseIndices.Select((c, idx) => (c, idx)).ToDictionary(x => x.c, x => x.idx);

        for (int i = 0; i < n; i++)
        {
            if (isCoarse[i])
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
                    if (isCoarse[col] && col != i)
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

        P = new SparseMatrixCSR(n, nCoarse, pValues.ToArray(), pColIndices.ToArray(), pRowPointers.ToArray());
        R = P.Transpose();

        // Step 3: Compute coarse matrix A_c = R * A * P
        SparseMatrixCSR ACoarse = R.MultiplyMatrix(A.MultiplyMatrix(P));

        // Create coarse level
        RugeStubenAMGLevel coarseLevel = new RugeStubenAMGLevel(ACoarse);
        coarseLevel.R = R;
        coarseLevel.P = P;
        return coarseLevel;
    }
}