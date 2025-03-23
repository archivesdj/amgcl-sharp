using Amgcl.Matrix;
using Amgcl.Solver;
using Amgcl.Enums;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda; // Assuming CUDA GPU

namespace AmgclXUnitTest.Solver;

public class GPUSolverTests
{
    [Fact]
    public void Test_SolveByAMGSolverGPU()
    {
        int gridSize = 30;
        int n = gridSize * gridSize;
        var matrix = Create2DPoissonMatrix(gridSize);
        double[] b = new double[n];
        for (int i = 0; i < n; i++) b[i] = 1.0;

         // Create a shared Accelerator instance
        using var context = Context.Create(builder => builder.Cuda());
        using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);


        ISolver solver = new AMGSolverGPU(matrix, accelerator);
        if (solver is AMGSolverGPU amgSolver)
            {
                //amgSolver.CoarseningType = CoarseningType.SmoothedAggregation;
                //amgSolver.MaxLevels = 4;
                amgSolver.SmootherType = SolverType.DampedJacobiGPU;
                //continue;
                amgSolver.BuildHierarchy();
            }
        double[] x = solver.Solve(b, 100, 1e-6);

        Console.WriteLine("Solution (first 5): " + string.Join(" ", x.Take(5).Select(v => v.ToString("F4"))));
        double[] Ax = new double[n];
        matrix.Multiply(x, Ax);
        Console.WriteLine("Computed Ax (first 5): " + string.Join(" ", Ax.Take(5).Select(v => v.ToString("F4"))));
    }

    [Fact(Skip = "done")]
    public void Test_SolveByLUDirectSolverGPU()
    {
        int gridSize = 10;
        int n = gridSize * gridSize;
        var matrix = Create2DPoissonMatrix(gridSize);
        double[] b = new double[n];
        for (int i = 0; i < n; i++) b[i] = 1.0;

        // Create a shared Accelerator instance
        using var context = Context.Create(builder => builder.Cuda());
        using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        // Test LUDirectSolverGPU
        var luSolver = new LUDirectSolverGPU(matrix, accelerator);
        double[] x = luSolver.Solve(b, 1, 1e-6); // maxIterations and tolerance ignored in direct solve
        Console.WriteLine("LUDirectSolverGPU Solution (first 5): " + string.Join(" ", x.Take(5).Select(v => v.ToString("F4"))));
        double[] Ax = new double[n];
        matrix.Multiply(x, Ax);
        Console.WriteLine("Computed Ax (first 5): " + string.Join(" ", Ax.Take(5).Select(v => v.ToString("F4"))));
    }

    [Fact(Skip = "done")]
    public void Test_SolveByGaussSeidelSolverGPU()
    {
        int gridSize = 10;
        int n = gridSize * gridSize;
        var matrix = Create2DPoissonMatrix(gridSize); // Your Poisson matrix function
        double[] b = new double[n];
        for (int i = 0; i < n; i++) b[i] = 1.0;

        using var context = Context.Create(builder => builder.Cuda());
        using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);
        var solver = new GaussSeidelSolverGPU(matrix, accelerator);
        double[] x = solver.Solve(b, 100, 1e-6);

        Console.WriteLine("GaussSeidelSolverGPU Solution (first 5): " + string.Join(" ", x.Take(5).Select(v => v.ToString("F4"))));
        double[] Ax = new double[n];
        matrix.Multiply(x, Ax);
        Console.WriteLine("Computed Ax (first 5): " + string.Join(" ", Ax.Take(5).Select(v => v.ToString("F4"))));
    }

    [Fact(Skip = "done")]
    public void Test_SolveByDampedJacobiSolverGPU()
    {
        int gridSize = 10;
        int n = gridSize * gridSize;
        var matrix = Create2DPoissonMatrix(gridSize); // Your Poisson matrix function
        double[] b = new double[n];
        for (int i = 0; i < n; i++) b[i] = 1.0;

        using var context = Context.Create(builder => builder.Cuda());
        using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);
        var solver = new DampedJacobiSolverGPU(matrix, accelerator, omega: 0.8);
        double[] x = solver.Solve(b, 100, 1e-6);

        Console.WriteLine("DampedJacobiSolverGPU Solution (first 5): " + string.Join(" ", x.Take(5).Select(v => v.ToString("F4"))));
        double[] Ax = new double[n];
        matrix.Multiply(x, Ax);
        Console.WriteLine("Computed Ax (first 5): " + string.Join(" ", Ax.Take(5).Select(v => v.ToString("F4"))));
    }
    //

    // Create2DPoissonMatrix (unchanged from previous examples)
    static SparseMatrixCSR Create2DPoissonMatrix(int gridSize)
    {
        int n = gridSize * gridSize;
        var values = new List<double>();
        var colIndices = new List<int>();
        var rowPointers = new List<int> { 0 };

        for (int i = 0; i < n; i++)
        {
            int row = i / gridSize;
            int col = i % gridSize;

            values.Add(4.0); colIndices.Add(i); // Diagonal
            if (col > 0) { values.Add(-1.0); colIndices.Add(i - 1); } // Left
            if (col < gridSize - 1) { values.Add(-1.0); colIndices.Add(i + 1); } // Right
            if (row > 0) { values.Add(-1.0); colIndices.Add(i - gridSize); } // Up
            if (row < gridSize - 1) { values.Add(-1.0); colIndices.Add(i + gridSize); } // Down

            rowPointers.Add(values.Count);
        }

        return new SparseMatrixCSR(n, n, values.ToArray(), colIndices.ToArray(), rowPointers.ToArray());
    }
}