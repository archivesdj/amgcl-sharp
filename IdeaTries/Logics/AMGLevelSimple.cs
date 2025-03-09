namespace AmgSharp.Logics;

public class SimpleAMGLevel : IAMGLevel
{
    public SparseMatrix A { get; private set; }        // 현재 레벨 행렬 (필수 초기화)
    public SparseMatrix? P { get; private set; }       // Interpolation 행렬 (초기화 전 null 가능)
    public SparseMatrix? R { get; private set; }       // Restriction 행렬 (초기화 전 null 가능)
    public SparseMatrix? CoarseA { get; private set; } // 다음 레벨 행렬 (초기화 전 null 가능)
    public int CoarseSize { get; private set; }        // 굵은 격자 크기

    public SimpleAMGLevel(SparseMatrix matrix)
    {
        if (matrix == null || matrix.Rows == 0 || matrix.Cols == 0)
            throw new ArgumentException("Invalid SparseMatrix input.");
        A = matrix;
        CoarseSize = 0;
        // P, R, CoarseA는 BuildCoarseLevel에서 초기화되므로 여기서 null로 남김
    }

    public void BuildCoarseLevel(double strengthThreshold = 0.25, double dampingFactor = 0.5)
    {
        CoarseSize = A.Rows / 2;
        if (CoarseSize <= 0)
            throw new InvalidOperationException("Coarse size too small.");

        // P 행렬 생성
        P = BuildInterpolationMatrix();
        if (P == null)
            throw new InvalidOperationException("Failed to build interpolation matrix.");

        // R = P^T
        R = Transpose(P);
        if (R == null)
            throw new InvalidOperationException("Failed to transpose matrix.");

        // CoarseA = R * A * P
        CoarseA = ComputeRAP(A, R, P);
        if (CoarseA == null)
            throw new InvalidOperationException("Failed to compute coarse matrix.");
    }

    private SparseMatrix? BuildInterpolationMatrix()
    {
        try
        {
            int fineSize = A.Rows;
            var pValues = new List<double>();
            var pColIndices = new List<int>();
            var pRowPointers = new List<int> { 0 };

            for (int i = 0; i < fineSize; i++)
            {
                pValues.Add(1.0);
                pColIndices.Add(i / 2);
                pRowPointers.Add(pRowPointers[^1] + 1);
            }

            return new SparseMatrix(fineSize, CoarseSize, pValues.ToArray(), pColIndices.ToArray(), pRowPointers.ToArray());
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in BuildInterpolationMatrix: {ex.Message}");
            return null;
        }
    }

    private static SparseMatrix? Transpose(SparseMatrix m)
    {
        try
        {
            var cols = new List<List<(int row, double val)>>();
            for (int i = 0; i < m.Cols; i++) cols.Add(new List<(int, double)>());
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = m.RowPointers[i]; j < m.RowPointers[i + 1]; j++)
                {
                    cols[m.ColIndices[j]].Add((i, m.Values[j]));
                }
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

    private static SparseMatrix? ComputeRAP(SparseMatrix a, SparseMatrix r, SparseMatrix p)
    {
        try
        {
            var values = new List<double>();
            var colIndices = new List<int>();
            var rowPointers = new List<int> { 0 };

            for (int i = 0; i < r.Rows; i++)
            {
                var rowR = ExtractRow(r, i);
                for (int j = 0; j < p.Cols; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < a.Rows; k++)
                    {
                        if (rowR.ContainsKey(k))
                        {
                            var rowA = ExtractRow(a, k);
                            for (int m = 0; m < p.Cols; m++)
                            {
                                if (m == j && rowA.ContainsKey(m))
                                    sum += rowR[k] * rowA[m];
                            }
                        }
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

    private static Dictionary<int, double> ExtractRow(SparseMatrix m, int row)
    {
        var result = new Dictionary<int, double>();
        for (int j = m.RowPointers[row]; j < m.RowPointers[row + 1]; j++)
        {
            result[m.ColIndices[j]] = m.Values[j];
        }
        return result;
    }
}