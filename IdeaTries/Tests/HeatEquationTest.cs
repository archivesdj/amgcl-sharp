public class HeatEquationTest
{
    public static void Run()
    {
        try
        {
            GenerateHeatEquationMatrix("heat16x16.mtx", 16);
            SparseMatrix A = SparseMatrix.FromMatrixMarket("heat16x16.mtx");
            double[] b = GenerateRightHandSide(16);

            var solver = new AMGSolver(A);
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            double[] x = solver.Solve(b, maxIterations: 200, tolerance: 1e-8);
            Console.WriteLine($"Solution completed in {stopwatch.Elapsed.TotalSeconds} seconds.");

            Console.WriteLine("First 5 elements of solution:");
            for (int i = 0; i < Math.Min(5, x.Length); i++)
                Console.WriteLine($"x[{i}] = {x[i]}");

            double[] r = Subtract(b, A.Multiply(x));
            Console.WriteLine($"Residual norm: {Norm(r)}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    static void GenerateHeatEquationMatrix(string filePath, int gridSize)
    {
        int n = gridSize * gridSize;
        int boundaryNodes = 4 * gridSize - 4; // 상하좌우 경계 - 모서리 중복
        int internalNodes = n - boundaryNodes;
        int nnz = boundaryNodes + 5 * internalNodes; // 경계(1개씩) + 내부(5개씩)

        using (StreamWriter writer = new StreamWriter(filePath))
        {
            writer.WriteLine("%%MatrixMarket matrix coordinate real symmetric");
            writer.WriteLine($"{n} {n} {nnz}");
            // 나머지 행렬 생성 로직은 동일
            for (int i = 0; i < gridSize; i++)
            {
                for (int j = 0; j < gridSize; j++)
                {
                    int idx = i * gridSize + j + 1;
                    bool isBoundary = (i == 0 || i == gridSize - 1 || j == 0 || j == gridSize - 1);

                    if (isBoundary)
                        writer.WriteLine($"{idx} {idx} 1.0");
                    else
                    {
                        writer.WriteLine($"{idx} {idx} 4.0");
                        writer.WriteLine($"{idx} {idx - 1} -1.0");
                        writer.WriteLine($"{idx} {idx + 1} -1.0");
                        writer.WriteLine($"{idx} {idx - gridSize} -1.0");
                        writer.WriteLine($"{idx} {idx + gridSize} -1.0");
                    }
                }
            }
        }
    }

    static double[] GenerateRightHandSide(int gridSize)
    {
        int n = gridSize * gridSize;
        double h = 1.0 / (gridSize - 1);
        double[] b = new double[n];

        for (int i = 0; i < gridSize; i++)
        {
            for (int j = 0; j < gridSize; j++)
            {
                int idx = i * gridSize + j;
                double x = j * h;
                double y = i * h;

                if (i == 0) b[idx] = 0.0; // 하단 경계
                else if (i == gridSize - 1) b[idx] = 1.0; // 상단 경계
                else if (j == 0 || j == gridSize - 1) b[idx] = 0.0; // 좌우 경계
                else b[idx] = Math.Sin(Math.PI * x) * Math.Sin(Math.PI * y) * h * h; // 내부 소스 항
            }
        }
        return b;
    }

    static double[] Subtract(double[] a, double[] b) => a.Zip(b, (x, y) => x - y).ToArray();
    static double Norm(double[] v) => Math.Sqrt(v.Sum(x => x * x));
}