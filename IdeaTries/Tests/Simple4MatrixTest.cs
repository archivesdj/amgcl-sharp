using AmgSharp.Logics;

namespace AmgSharp.Tests;

public class Simple4MatrixTest
{
    public static void Run()
    {
        try
        {
            Generate4x4LaplaceMatrix("laplace4x4.mtx");
            SparseMatrix A = SparseMatrix.FromMatrixMarket("laplace4x4.mtx");
            double[] b = GenerateRightHandSide(4);

            using var solver = new AMGSolver(A, maxLevels: 3);
            double[] x = solver.Solve(b, maxIterations: 10, tolerance: 1e-8);

            Console.WriteLine("Solution:");
            for (int i = 0; i < x.Length; i++)
                Console.WriteLine($"x[{i}] = {x[i]}");

            double[] r = AMGSolver.Subtract(b, A.Multiply(x));
            Console.WriteLine($"Residual norm: {AMGSolver.Norm(r)}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    static void Generate4x4LaplaceMatrix(string filePath)
    {
        using (StreamWriter writer = new StreamWriter(filePath))
        {
            writer.WriteLine("%%MatrixMarket matrix coordinate real symmetric");
            writer.WriteLine("4 4 10"); // 4x4, nnz=10
            writer.WriteLine("1 1 2.0");
            writer.WriteLine("1 2 -1.0");
            writer.WriteLine("2 1 -1.0");
            writer.WriteLine("2 2 2.0");
            writer.WriteLine("2 3 -1.0");
            writer.WriteLine("3 2 -1.0");
            writer.WriteLine("3 3 2.0");
            writer.WriteLine("3 4 -1.0");
            writer.WriteLine("4 3 -1.0");
            writer.WriteLine("4 4 2.0");
        }
    }

    static double[] GenerateRightHandSide(int n)
    {
        double[] b = new double[n];
        for (int i = 0; i < n; i++)
            b[i] = 1.0; // f(x) = 1
        double actualNorm = AMGSolver.Norm(b);
        double expectedNorm = Math.Sqrt(n);
        Console.WriteLine($"Expected b norm = {expectedNorm}, Actual b norm = {actualNorm}");
        if (Math.Abs(actualNorm - expectedNorm) > 1e-10)
            throw new Exception($"b norm mismatch! Expected {expectedNorm}, got {actualNorm}");
        return b;
    }
}