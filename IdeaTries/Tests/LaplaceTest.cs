using AmgSharp.Logics;

namespace AmgSharp.Tests;

public class LaplaceTest
{
    public static void Run()
    {
        try
        {
            Generate1DLaplaceMatrix("laplace16.mtx", 16);
            SparseMatrix A = SparseMatrix.FromMatrixMarket("laplace16.mtx");
            double[] b = GenerateRightHandSide(16);

            using var solver = new AMGSolver(A, maxLevels: 10);
            double[] x = solver.Solve(b, maxIterations: 200, tolerance: 1e-8);

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

    static void Generate1DLaplaceMatrix(string filePath, int n)
    {
        int nnz = 3 * n - 2;
        using (StreamWriter writer = new StreamWriter(filePath))
        {
            writer.WriteLine("%%MatrixMarket matrix coordinate real symmetric");
            writer.WriteLine($"{n} {n} {nnz}");

            for (int i = 1; i <= n; i++)
            {
                writer.WriteLine($"{i} {i} 2.0");
                if (i > 1) writer.WriteLine($"{i} {i-1} -1.0");
                if (i < n) writer.WriteLine($"{i} {i+1} -1.0");
            }
        }
    }

    static double[] GenerateRightHandSide(int n)
    {
        double h = 1.0 / (n + 1);
        double[] b = new double[n];
        for (int i = 0; i < n; i++)
            b[i] = h * h; // f(x) = 1, h^2 스케일링
        double actualNorm = AMGSolver.Norm(b);
        double expectedNorm = h * Math.Sqrt(n);
        Console.WriteLine($"h = {h}, h^2 = {h * h}, Expected b norm = {expectedNorm}, Actual b norm = {actualNorm}");
        if (Math.Abs(actualNorm - expectedNorm) > 1e-10)
            Console.WriteLine("Warning: b norm mismatch!");
        return b;
    }
}