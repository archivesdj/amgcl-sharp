using ILGPU.Runtime;

namespace AmgSharp.Logics;

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
            throw new ArgumentException("Invalid input matrix.");
        A = matrix;
        CoarseSize = 0;
        //Console.WriteLine($"A: nnz = {A.Values.Length}, Values = [{string.Join(", ", A.Values)}], ColIndices = [{string.Join(", ", A.ColIndices)}], RowPointers = [{string.Join(", ", A.RowPointers)}]");
    }

    public void BuildCoarseLevel(double strengthThreshold = 0.9, double dampingFactor = 0.8)
    {
        try
        {
            var strongConnections = ComputeStrongConnections(strengthThreshold);
            //Console.WriteLine($"Strong connections computed for {A.Rows} nodes. Total connections: {strongConnections.Sum(s => s.Count)}");

            var aggregates = AggregateNodes(strongConnections);
            //Console.WriteLine($"CoarseSize set to {CoarseSize} from {A.Rows} (Target: {A.Rows / 2})");

            P = BuildInterpolationMatrix(aggregates, dampingFactor);
            if (P == null) throw new InvalidOperationException("Failed to create interpolation matrix (P).");
            //Console.WriteLine($"P created: {P.Rows}x{P.Cols}, nnz: {P.Values.Length}");

            R = Transpose(P);
            if (R == null) throw new InvalidOperationException("Failed to create restriction matrix (R).");
            //Console.WriteLine($"R created: {R.Rows}x{R.Cols}, nnz: {R.Values.Length}");

            CoarseA = ComputeRAP(A, R, P);
            if (CoarseA == null) throw new InvalidOperationException("Failed to create coarse matrix (CoarseA).");
            //Console.WriteLine($"CoarseA created: {CoarseA.Rows}x{CoarseA.Cols}, nnz: {CoarseA.Values.Length}");

            Console.WriteLine($"Coarse level built: {A.Rows} → {CoarseSize}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in BuildCoarseLevel: {ex.Message}");
            throw;
        }
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
            if (maxOffDiag == 0) maxOffDiag = diag[i] * 0.1;

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
        int targetCoarseSize = Math.Max(n / 2, 1);
        var unaggregated = new HashSet<int>(Enumerable.Range(0, n));
        int coarseIndex = 0;

        //Console.WriteLine($"Strong connections: [{string.Join(", ", strongConnections.Select(s => $"{{{string.Join(", ", s)}}}"))}]");

        // Laplace 행렬의 순차적 쌍(0-1, 2-3)을 강제 적용
        while (coarseIndex < targetCoarseSize && unaggregated.Count > 1)
        {
            int seed = unaggregated.First(); // 첫 번째 노드부터 순차적으로
            aggregates[seed] = coarseIndex;
            unaggregated.Remove(seed);

            // 인접 노드 중 강한 연결이 있는 첫 번째 노드 선택
            var neighbor = strongConnections[seed]
                .Where(j => unaggregated.Contains(j))
                .FirstOrDefault();

            if (neighbor != default(int))
            {
                aggregates[neighbor] = coarseIndex;
                unaggregated.Remove(neighbor);
            }
            coarseIndex++;
        }

        // 남은 노드 처리
        foreach (int i in unaggregated.ToList())
        {
            aggregates[i] = aggregates[strongConnections[i].FirstOrDefault(j => aggregates[j] != -1)];
            if (aggregates[i] == -1) // 연결된 노드가 없으면 새 집합
                aggregates[i] = coarseIndex < targetCoarseSize ? coarseIndex++ : 0;
            unaggregated.Remove(i);
        }

        CoarseSize = coarseIndex;
        if (CoarseSize != targetCoarseSize)
            Console.WriteLine($"Adjusted CoarseSize to {CoarseSize} from {coarseIndex} to match target {targetCoarseSize}");

        //Console.WriteLine($"Aggregates = [{string.Join(", ", aggregates)}]");
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

            double[] diag = new double[A.Rows];
            for (int i = 0; i < A.Rows; i++)
            {
                for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
                {
                    if (A.ColIndices[j] == i)
                    {
                        diag[i] = A.Values[j] != 0 ? 1.0 / A.Values[j] : 1.0;
                        break;
                    }
                }
            }

            for (int iter = 0; iter < 3; iter++)
            {
                var smoothedValues = new List<double>();
                var smoothedColIndices = new List<int>();
                var smoothedRowPointers = new List<int> { 0 };
                double[] ap = A.Multiply(P.Multiply(new double[CoarseSize]));

                for (int i = 0; i < fineSize; i++)
                {
                    for (int j = P.RowPointers[i]; j < P.RowPointers[i + 1]; j++)
                    {
                        int col = P.ColIndices[j];
                        double value = P.Values[j] - dampingFactor * diag[i] * ap[i];
                        if (Math.Abs(value) > 1e-10)
                        {
                            smoothedValues.Add(value);
                            smoothedColIndices.Add(col);
                        }
                    }
                    smoothedRowPointers.Add(smoothedValues.Count);
                }
                P = new SparseMatrix(fineSize, CoarseSize, smoothedValues.ToArray(), smoothedColIndices.ToArray(), smoothedRowPointers.ToArray());
            }

            //Console.WriteLine($"P: nnz = {P.Values.Length}, Values = [{string.Join(", ", P.Values)}], ColIndices = [{string.Join(", ", P.ColIndices)}], RowPointers = [{string.Join(", ", P.RowPointers)}]");
            return P;
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
            var values = new List<double>();
            var colIndices = new List<int>();
            var rowPointers = new List<int> { 0 };
            var temp = new List<(int row, double val)>[m.Cols];
            for (int i = 0; i < m.Cols; i++) temp[i] = new List<(int, double)>();

            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = m.RowPointers[i]; j < m.RowPointers[i + 1]; j++)
                    temp[m.ColIndices[j]].Add((i, m.Values[j]));
            }

            for (int i = 0; i < m.Cols; i++)
            {
                double weight = 0.5; // 가중 평균으로 조정
                foreach (var (row, val) in temp[i])
                {
                    values.Add(val * weight);
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
            // A * P 계산
            var apRows = new List<(double Value, int Col)>[a.Rows];
            for (int i = 0; i < a.Rows; i++)
            {
                var rowContribs = new Dictionary<int, double>();
                //Console.Write($"Row {i}: ");
                for (int k = a.RowPointers[i]; k < a.RowPointers[i + 1]; k++)
                {
                    int aCol = a.ColIndices[k];
                    double aVal = a.Values[k];
                    for (int l = p.RowPointers[aCol]; l < p.RowPointers[aCol + 1]; l++)
                    {
                        int pCol = p.ColIndices[l];
                        if (!rowContribs.ContainsKey(pCol))
                            rowContribs[pCol] = 0;
                        rowContribs[pCol] += aVal * p.Values[l];
                        //Console.Write($"{aVal} * P[{aCol},{pCol}] + ");
                    }
                }
                //Console.WriteLine($"= [{string.Join(", ", rowContribs.Select(kv => $"{kv.Key}:{kv.Value}"))}]");
                apRows[i] = rowContribs.Select(kv => (kv.Value, kv.Key)).Where(t => Math.Abs(t.Value) > 1e-10).ToList();
            }

            var apValues = new List<double>();
            var apColIndices = new List<int>();
            var apRowPointers = new List<int> { 0 };
            foreach (var row in apRows)
            {
                foreach (var (val, col) in row.OrderBy(t => t.Col))
                {
                    apValues.Add(val);
                    apColIndices.Add(col);
                }
                apRowPointers.Add(apValues.Count);
            }
            //Console.WriteLine($"Before SparseMatrix: Values = [{string.Join(", ", apValues)}], ColIndices = [{string.Join(", ", apColIndices)}], RowPointers = [{string.Join(", ", apRowPointers)}]");
            var ap = new SparseMatrix(a.Rows, p.Cols, apValues.ToArray(), apColIndices.ToArray(), apRowPointers.ToArray());
            //Console.WriteLine($"A * P: nnz = {ap.Values.Length}, Sum = {ap.Values.Sum()}, Values = [{string.Join(", ", ap.Values)}], ColIndices = [{string.Join(", ", ap.ColIndices)}]");

            // R * (A * P)
            var values = new List<double>();
            var colIndices = new List<int>();
            var rowPointers = new List<int> { 0 };

            for (int i = 0; i < r.Rows; i++)
            {
                var rowContribs = new Dictionary<int, double>();
                for (int k = r.RowPointers[i]; k < r.RowPointers[i + 1]; k++)
                {
                    int rCol = r.ColIndices[k];
                    for (int l = ap.RowPointers[rCol]; l < ap.RowPointers[rCol + 1]; l++)
                    {
                        int apCol = ap.ColIndices[l];
                        if (!rowContribs.ContainsKey(apCol))
                            rowContribs[apCol] = 0;
                        rowContribs[apCol] += r.Values[k] * ap.Values[l];
                    }
                }
                foreach (var kv in rowContribs.OrderBy(k => k.Key))
                {
                    if (Math.Abs(kv.Value) > 1e-10)
                    {
                        values.Add(kv.Value);
                        colIndices.Add(kv.Key);
                    }
                }
                rowPointers.Add(values.Count);
            }

            var coarseA = new SparseMatrix(r.Rows, p.Cols, values.ToArray(), colIndices.ToArray(), rowPointers.ToArray());
            double diagSum = 0;
            for (int i = 0; i < coarseA.Rows; i++)
                for (int j = coarseA.RowPointers[i]; j < coarseA.RowPointers[i + 1]; j++)
                    if (coarseA.ColIndices[j] == i)
                        diagSum += coarseA.Values[j];
            //Console.WriteLine($"RAP computed: nnz = {coarseA.Values.Length}, Sum of values = {coarseA.Values.Sum()}, Diagonal sum = {diagSum}, Values = [{string.Join(", ", coarseA.Values)}]");
            return coarseA;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in ComputeRAP: {ex.Message}");
            return null;
        }
    }
}