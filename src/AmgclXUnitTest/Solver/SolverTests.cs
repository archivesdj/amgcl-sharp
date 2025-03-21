using Amgcl.Matrix;
using Amgcl.Solver;

namespace AmgclXUnitTest.Solver;

public class DampedJacobiSolverTests
{
    [Fact(Skip = "done")]
    public void Test_SolveSymmetricMatrix()
    {
        double[,] dense = new double[,]
        {
            { 4.0, 1.0, 0.0 },
            { 1.0, 4.0, 1.0 },
            { 0.0, 1.0, 4.0 }
        };
        var matrix = SparseMatrixCSR.FromDense(dense);
        double[] b = { 5.0, 6.0, 5.0 };
        double[] xExpected = { 1.0, 1.0, 1.0 };
        Console.WriteLine(matrix.Print());

        // Damped Jacobi
        ISolver jacobi = new DampedJacobiSolver(matrix) { Omega = 0.8 };
        double[] xJacobi = jacobi.Solve(b, 1000, 1e-6);
        Console.WriteLine("Damped Jacobi Solution:");
        Console.WriteLine(string.Join(" ", xJacobi.Select(v => v.ToString("F4"))));
        Assert.Equal(xExpected, xJacobi.Select(v => Math.Round(v, 4)).ToArray());

        // Conjugate Gradient
        ISolver cg = new ConjugateGradientSolver(matrix);
        double[] xCG = cg.Solve(b, 1000, 1e-6);
        Console.WriteLine("Conjugate Gradient Solution:");
        Console.WriteLine(string.Join(" ", xCG.Select(v => v.ToString("F4"))));
        Assert.Equal(xExpected, xCG.Select(v => Math.Round(v, 4)).ToArray());

        // Bi Conjugate Gradient Stabilized
        ISolver biCGSTAB = new BiCGSTABSolver(matrix);
        double[] xBiCGSTAB = biCGSTAB.Solve(b, 1000, 1e-6);
        Console.WriteLine("Bi Conjugate Gradient Stabilized Solution:");
        Console.WriteLine(string.Join(" ", xBiCGSTAB.Select(v => v.ToString("F4"))));
        Assert.Equal(xExpected, xBiCGSTAB.Select(v => Math.Round(v, 4)).ToArray());

        // Bi Conjugate Gradient Stabilized
        ISolver lud = new LUDirectSolver(matrix);
        double[] xLUD = lud.Solve(b, 1000, 1e-6);
        Console.WriteLine("LU Direct Solution:");
        Console.WriteLine(string.Join(" ", xLUD.Select(v => v.ToString("F4"))));
        Assert.Equal(xExpected, xLUD.Select(v => Math.Round(v, 4)).ToArray());
    }

    [Fact(Skip = "done")]
    public void Test_SolveAsymmetricMatrix()
    {
        // Define a 5x5 asymmetric sparse matrix
        double[,] dense = new double[,]
        {
            { 4.0, -1.0,  0.0,  0.0,  0.0 },
            {-1.0,  4.0, -1.0,  0.0,  0.0 },
            { 0.0, -1.0,  4.0, -1.0,  0.0 },
            { 0.0,  0.0, -1.0,  4.0, -2.0 },
            {-1.0,  0.0,  0.0, -2.0,  5.0 }
        };
        var matrix = SparseMatrixCSR.FromDense(dense);

        // Define the right-hand side vector b
        double[] b = { 3.0, 2.0, 1.0, 0.0, 2.0 };

        double[] xExpected = { 0.9744,0.8974,0.6154,0.5641,0.8205 };

        // Print the matrix A
        Console.WriteLine("Matrix A:");
        Console.WriteLine(matrix.Print());

        Console.WriteLine("BiCGSTAB Solution (x):");
        ISolver biCGSTABSolver = new BiCGSTABSolver(matrix);
        double[] xBiCGSTAB = biCGSTABSolver.Solve(b, 1000, 1e-6);
        Console.WriteLine(string.Join(" ", xBiCGSTAB.Select(v => v.ToString("F4"))));
        Assert.Equal(xExpected, xBiCGSTAB.Select(v => Math.Round(v, 4)).ToArray());

        // Verify the result by computing Ax
        //double[] Ax = new double[matrix.Rows];
        //matrix.Multiply(xBiCGSTAB, Ax);
        //Console.WriteLine("Computed Ax:");
        //Console.WriteLine(string.Join(" ", Ax.Select(v => v.ToString("F4"))));
        //Assert.Equal(b, Ax.Select(v => Math.Round(v, 4)).ToArray());

        Console.WriteLine("LU Direct Solution (x):");
        ISolver lud = new LUDirectSolver(matrix);
        double[] xLUD = lud.Solve(b, 1000, 1e-6);
        Console.WriteLine(string.Join(" ", xLUD.Select(v => v.ToString("F4"))));
        Assert.Equal(xExpected, xLUD.Select(v => Math.Round(v, 4)).ToArray());

        Console.WriteLine("Gauss-Seidel Solution (x):");
        ISolver gs = new GaussSeidelSolver(matrix);
        double[] xGS = gs.Solve(b, 1000, 1e-6);
        Console.WriteLine(string.Join(" ", xGS.Select(v => v.ToString("F4"))));
        Assert.Equal(xExpected, xLUD.Select(v => Math.Round(v, 4)).ToArray());
    }
}