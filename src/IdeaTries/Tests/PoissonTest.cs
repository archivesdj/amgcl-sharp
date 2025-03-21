using AmgSharp.Logics;

namespace AmgSharp.Tests;

public class PoissonTest {
    public static void Run()
    {
        try
        {
            GeneratePoissonMatrix("poisson16x16.mtx", 16);
            SparseMatrix A = SparseMatrix.FromMatrixMarket("poisson16x16.mtx");
            double[] b = new double[A.Rows];
            Array.Fill(b, 1.0);

            var solver = new AMGSolver(A, maxLevels: 10);
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            double[] x = solver.Solve(b);
            Console.WriteLine($"Solution completed in {stopwatch.Elapsed.TotalSeconds} seconds.");

            Console.WriteLine("First 5 elements of solution:");
            for (int i = 0; i < Math.Min(5, x.Length); i++)
                Console.WriteLine($"x[{i}] = {x[i]}");

            double[] r = AMGSolver.Subtract(b, A.Multiply(x));
            Console.WriteLine($"Residual norm: {AMGSolver.Norm(r)}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

     static void GeneratePoissonMatrix(string filePath, int gridSize)
    {
        int n = gridSize * gridSize;
        using (StreamWriter writer = new StreamWriter(filePath))
        {
            writer.WriteLine("%%MatrixMarket matrix coordinate real symmetric");
            writer.WriteLine($"{n} {n} {5 * n - 4 * gridSize}");
            for (int i = 0; i < gridSize; i++)
            {
                for (int j = 0; j < gridSize; j++)
                {
                    int idx = i * gridSize + j + 1;
                    writer.WriteLine($"{idx} {idx} 4.0");
                    if (j > 0) writer.WriteLine($"{idx} {idx - 1} -1.0");
                    if (j < gridSize - 1) writer.WriteLine($"{idx} {idx + 1} -1.0");
                    if (i > 0) writer.WriteLine($"{idx} {idx - gridSize} -1.0");
                    if (i < gridSize - 1) writer.WriteLine($"{idx} {idx + gridSize} -1.0");
                }
            }
        }
    }
}