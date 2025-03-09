namespace AmgSharp.Logics;

public class SparseMatrix
{
    public int Rows { get; private set; }
    public int Cols { get; private set; }
    public double[] Values { get; private set; }
    public int[] ColIndices { get; private set; }
    public int[] RowPointers { get; private set; }

    public SparseMatrix(int rows, int cols, double[] values, int[] colIndices, int[] rowPointers)
    {
        Rows = rows;
        Cols = cols;
        Values = values;
        ColIndices = colIndices;
        RowPointers = rowPointers;
        //Console.WriteLine($"SparseMatrix created: Values = [{string.Join(", ", Values)}], ColIndices = [{string.Join(", ", ColIndices)}], RowPointers = [{string.Join(", ", RowPointers)}]");
    }

    // Matrix-vector multiplication
    public double[] Multiply(double[] x)
    {
        double[] result = new double[Rows];
        for (int i = 0; i < Rows; i++)
        {
            double sum = 0;
            for (int j = RowPointers[i]; j < RowPointers[i + 1]; j++)
            {
                sum += Values[j] * x[ColIndices[j]];
            }
            result[i] = sum;
        }
        return result;
    }

    // Matrix Market 파일 읽기
    public static SparseMatrix FromMatrixMarket(string filePath)
    {
        string[] lines = File.ReadAllLines(filePath);
        int lineIndex = 0;

        // 헤더 건너뛰기
        while (lineIndex < lines.Length && lines[lineIndex].StartsWith("%")) lineIndex++;

        // 크기 정보 파싱
        if (lineIndex >= lines.Length) throw new Exception("Invalid Matrix Market file: size info missing.");
        string[] sizeInfo = lines[lineIndex].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (sizeInfo.Length < 3) throw new Exception("Invalid size info.");
        int rows = int.Parse(sizeInfo[0]);
        int cols = int.Parse(sizeInfo[1]);
        int nnz = int.Parse(sizeInfo[2]);
        lineIndex++;

        // COO 형식으로 데이터 읽기
        List<(int row, int col, double val)> coo = new List<(int, int, double)>();
        for (; lineIndex < lines.Length; lineIndex++)
        {
            if (string.IsNullOrWhiteSpace(lines[lineIndex])) continue; // 빈 줄 무시
            string[] parts = lines[lineIndex].Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 3) continue; // 유효하지 않은 줄 무시
            int row = int.Parse(parts[0]) - 1; // 1-based -> 0-based
            int col = int.Parse(parts[1]) - 1; // 1-based -> 0-based
            double val = double.Parse(parts[2]);
            coo.Add((row, col, val));
        }

        if (coo.Count != nnz) throw new Exception($"Expected {nnz} non-zeros, but found {coo.Count}.");

        // COO를 행 순서로 정렬
        coo.Sort((a, b) => a.row == b.row ? a.col.CompareTo(b.col) : a.row.CompareTo(b.row));

        // CSR 변환
        double[] values = new double[nnz];
        int[] colIndices = new int[nnz];
        int[] rowPointers = new int[rows + 1];

        int currentIndex = 0;
        for (int i = 0; i < rows; i++)
        {
            rowPointers[i] = currentIndex;
            while (currentIndex < coo.Count && coo[currentIndex].row == i)
            {
                values[currentIndex] = coo[currentIndex].val;
                colIndices[currentIndex] = coo[currentIndex].col;
                currentIndex++;
            }
        }
        rowPointers[rows] = nnz; // 마지막 포인터 설정

        return new SparseMatrix(rows, cols, values, colIndices, rowPointers);
    }

    // Matrix Market 파일로 출력
    public void ToMatrixMarket(string filePath)
    {
        using (StreamWriter writer = new StreamWriter(filePath))
        {
            // 헤더 작성
            writer.WriteLine("%%MatrixMarket matrix coordinate real general");

            // 크기 정보 작성
            int nnz = Values.Length;
            writer.WriteLine($"{Rows} {Cols} {nnz}");

            // CSR 데이터를 COO 형식으로 변환하여 기록
            for (int i = 0; i < Rows; i++)
            {
                for (int j = RowPointers[i]; j < RowPointers[i + 1]; j++)
                {
                    int row = i + 1;              // 0-based -> 1-based
                    int col = ColIndices[j] + 1;  // 0-based -> 1-based
                    double val = Values[j];
                    writer.WriteLine($"{row} {col} {val}");
                }
            }
        }
    }
}