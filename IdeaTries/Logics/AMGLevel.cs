public class AMGLevel : IAMGLevel
{
    public SparseMatrix A { get; private set; }
    public SparseMatrix? P { get; private set; }
    public SparseMatrix? R { get; private set; }
    public SparseMatrix? CoarseA { get; private set; }
    public int CoarseSize { get; private set; }

    public AMGLevel(SparseMatrix matrix)
    {
        if (matrix == null || matrix.Rows == 0 || matrix.Cols == 0)
            throw new ArgumentException("Invalid SparseMatrix input.");
        A = matrix;
        CoarseSize = 0;
    }

    public void BuildCoarseLevel(double strengthThreshold = 0.05, double dampingFactor = 0.5)
    {
        var strongConnections = ComputeStrongConnections(strengthThreshold);
        var aggregates = AggregateNodes(strongConnections);
        P = BuildInterpolationMatrix(aggregates, dampingFactor);
        if (P == null) throw new InvalidOperationException("Failed to build interpolation matrix.");
        R = Transpose(P);
        if (R == null) throw new InvalidOperationException("Failed to transpose matrix.");
        CoarseA = ComputeRAP(A, R, P);
        if (CoarseA == null) throw new InvalidOperationException("Failed to compute coarse matrix.");
        Console.WriteLine($"CoarseSize reduced to {CoarseSize} from {A.Rows}");
    }

    private List<HashSet<int>> ComputeStrongConnections(double threshold)
    {
        var strongConnections = new List<HashSet<int>>(A.Rows);
        for (int i = 0; i < A.Rows; i++)
            strongConnections.Add(new HashSet<int>());

        double[] diag = new double[A.Rows];
        for (int i = 0; i < A.Rows; i++)
        {
            for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
            {
                if (A.ColIndices[j] == i)
                {
                    diag[i] = Math.Abs(A.Values[j]);
                    break;
                }
            }
        }

        for (int i = 0; i < A.Rows; i++)
        {
            double maxOffDiag = 0;
            for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
            {
                if (A.ColIndices[j] != i)
                    maxOffDiag = Math.Max(maxOffDiag, Math.Abs(A.Values[j]));
            }
            if (maxOffDiag == 0) maxOffDiag = diag[i] * 0.05;

            for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
            {
                int col = A.ColIndices[j];
                if (col != i)
                {
                    double value = Math.Abs(A.Values[j]);
                    if (value >= threshold * maxOffDiag)
                    {
                        strongConnections[i].Add(col);
                        strongConnections[col].Add(i);
                    }
                }
            }
        }

        return strongConnections;
    }

    private int[] AggregateNodes(List<HashSet<int>> strongConnections)
    {
        int n = A.Rows;
        int[] aggregates = new int[n];
        Array.Fill(aggregates, -1);
        int coarseIndex = 0;

        var nodes = Enumerable.Range(0, n)
            .OrderByDescending(i => strongConnections[i].Count)
            .ToList();

        foreach (int i in nodes)
        {
            if (aggregates[i] == -1)
            {
                aggregates[i] = coarseIndex;
                var neighbors = strongConnections[i].Where(j => aggregates[j] == -1).ToList();
                if (neighbors.Count > 0) // 최소 1개 이상 연결 포함
                {
                    foreach (int j in neighbors)
                        aggregates[j] = coarseIndex;
                }
                coarseIndex++;
            }
        }

        for (int i = 0; i < n; i++)
        {
            if (aggregates[i] == -1)
                aggregates[i] = coarseIndex++;
        }

        CoarseSize = coarseIndex;
        return aggregates;
    }

    private SparseMatrix? BuildInterpolationMatrix(int[] aggregates, double dampingFactor)
    {
        try
        {
            int fineSize = A.Rows;
            CoarseSize = aggregates.Max() + 1;

            var pValues = new List<double>();
            var pColIndices = new List<int>();
            var pRowPointers = new List<int> { 0 };
            for (int i = 0; i < fineSize; i++)
            {
                pValues.Add(1.0);
                pColIndices.Add(aggregates[i]);
                pRowPointers.Add(pRowPointers[^1] + 1);
            }
            var P = new SparseMatrix(fineSize, CoarseSize, pValues.ToArray(), pColIndices.ToArray(), pRowPointers.ToArray());

            double[] dInv = new double[A.Rows];
            for (int i = 0; i < A.Rows; i++)
            {
                for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
                {
                    if (A.ColIndices[j] == i)
                    {
                        dInv[i] = 1.0 / A.Values[j];
                        break;
                    }
                }
            }

            var smoothedValues = new List<double>();
            var smoothedColIndices = new List<int>();
            var smoothedRowPointers = new List<int> { 0 };
            double[] ap = A.Multiply(P.Multiply(new double[CoarseSize]));

            for (int i = 0; i < fineSize; i++)
            {
                for (int j = P.RowPointers[i]; j < P.RowPointers[i + 1]; j++)
                {
                    int col = P.ColIndices[j];
                    double value = P.Values[j] - dampingFactor * dInv[i] * ap[i];
                    if (Math.Abs(value) > 1e-10)
                    {
                        smoothedValues.Add(value);
                        smoothedColIndices.Add(col);
                    }
                }
                smoothedRowPointers.Add(smoothedValues.Count);
            }

            return new SparseMatrix(fineSize, CoarseSize, smoothedValues.ToArray(), smoothedColIndices.ToArray(), smoothedRowPointers.ToArray());
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in BuildInterpolationMatrix: {ex.Message}");
            return null;
        }
    }

    private SparseMatrix? Transpose(SparseMatrix m)
    {
        try
        {
            var cols = new List<List<(int row, double val)>>();
            for (int i = 0; i < m.Cols; i++) cols.Add(new List<(int, double)>());
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = m.RowPointers[i]; j < m.RowPointers[i + 1]; j++)
                    cols[m.ColIndices[j]].Add((i, m.Values[j]));
            }

            var values = new List<double>();
            var colIndices = new List<int>();
            var rowPointers = new List<int> { 0 };
            for (int i = 0; i < m.Cols; i++)
            {
                foreach (var (row, val) in cols[i])
                {
                    values.Add(val);
                    colIndices.Add(row);
                }
                rowPointers.Add(values.Count);
            }
            return new SparseMatrix(m.Cols, m.Rows, values.ToArray(), colIndices.ToArray(), rowPointers.ToArray());
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in Transpose: {ex.Message}");
            return null;
        }
    }

    private SparseMatrix? ComputeRAP(SparseMatrix a, SparseMatrix r, SparseMatrix p)
    {
        try
        {
            var ap = new double[p.Cols][];
            for (int j = 0; j < p.Cols; j++)
            {
                double[] col = new double[p.Rows];
                for (int k = p.RowPointers[0]; k < p.RowPointers[p.Rows]; k++)
                    if (p.ColIndices[k] == j)
                        col[k - p.RowPointers[0]] = p.Values[k];
                ap[j] = a.Multiply(col);
            }

            var values = new List<double>();
            var colIndices = new List<int>();
            var rowPointers = new List<int> { 0 };

            for (int i = 0; i < r.Rows; i++)
            {
                for (int j = 0; j < p.Cols; j++)
                {
                    double sum = 0;
                    for (int k = r.RowPointers[i]; k < r.RowPointers[i + 1]; k++)
                    {
                        int row = r.ColIndices[k];
                        sum += r.Values[k] * ap[j][row];
                    }
                    if (Math.Abs(sum) > 1e-10)
                    {
                        values.Add(sum);
                        colIndices.Add(j);
                    }
                }
                rowPointers.Add(values.Count);
            }
            return new SparseMatrix(r.Rows, p.Cols, values.ToArray(), colIndices.ToArray(), rowPointers.ToArray());
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in ComputeRAP: {ex.Message}");
            return null;
        }
    }
}