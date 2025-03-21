using Amgcl.Matrix;
using Amgcl.Solver;
using Amgcl.Helpers;
using Amgcl.Enums;

namespace AmgclXUnitTest.Solver;

public class SolverFactoryTests
{
    [Fact(Skip = "done")]
    public void Test_CreateSolver()
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

        // Define right-hand side vector b
        double[] b = { 3.0, 2.0, 1.0, 0.0, 2.0 };

        double[] xExpected = { 0.9744,0.8974,0.6154,0.5641,0.8205 };

        // Array of solver types to test
        var solverTypes = Enum.GetValues(typeof(SolverType)) as SolverType[];

        foreach (var type in solverTypes!)
        {
            // Create solver using factory
            ISolver solver = SolverFactory.CreateSolver(type, matrix);
            if (solver is AMGSolver amgSolver)
            {
                amgSolver.CoarseningType = CoarseningType.Aggregation;
                //continue;
                amgSolver.BuildHierarchy();
            }
            double[] x = solver.Solve(b, 1000, 1e-6);

            // Print solver type and solution
            Console.WriteLine($"{type} Solution (x):");
            Console.WriteLine(string.Join(" ", x.Select(v => v.ToString("F4"))));
            Assert.Equal(xExpected, x.Select(v => Math.Round(v, 4)).ToArray());
        }
    }

    [Fact(Skip = "done")]
    public void Test_AMGSolver()
    {
        // Create a 10x10 grid Poisson problem (100x100 matrix)
        int gridSize = 10;
        int n = gridSize * gridSize; // Total number of unknowns
        SparseMatrixCSR matrix = Create2DPoissonMatrix(gridSize);
        double[] b = CreateRightHandSide(n);

        // Test all coarsening types
        CoarseningType[]? coarseningTypes = Enum.GetValues(typeof(CoarseningType)) as CoarseningType[];

        foreach (var type in coarseningTypes!)
        {
            AMGSolver solver = new AMGSolver(matrix);
            solver.CoarseningType = type;
            solver.MaxLevels = 4; // Adjust for larger problem
            solver.SmootherType = SolverType.GaussSeidel;
            solver.BuildHierarchy();

            double[] x = solver.Solve(b, 100, 1e-6);

            // Print results
            Console.WriteLine($"AMG with {type} Coarsening Solution (first 5 elements):");
            Console.WriteLine(string.Join(" ", x.Take(5).Select(v => v.ToString("F4"))));
            Console.WriteLine("... (remaining elements omitted for brevity)");

            // Verify solution
            double[] Ax = new double[n];
            matrix.Multiply(x, Ax);
            Console.WriteLine($"Computed Ax (first 5 elements):");
            Console.WriteLine(string.Join(" ", Ax.Take(5).Select(v => v.ToString("F4"))));
            Console.WriteLine("... (remaining elements omitted for brevity)");
            Console.WriteLine();
        }
    }

    // Create a 2D Poisson matrix for a gridSize x gridSize grid
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

            // Diagonal: 4
            values.Add(4.0);
            colIndices.Add(i);

            // Left neighbor (if not on left boundary)
            if (col > 0)
            {
                values.Add(-1.0);
                colIndices.Add(i - 1);
            }

            // Right neighbor (if not on right boundary)
            if (col < gridSize - 1)
            {
                values.Add(-1.0);
                colIndices.Add(i + 1);
            }

            // Up neighbor (if not on top boundary)
            if (row > 0)
            {
                values.Add(-1.0);
                colIndices.Add(i - gridSize);
            }

            // Down neighbor (if not on bottom boundary)
            if (row < gridSize - 1)
            {
                values.Add(-1.0);
                colIndices.Add(i + gridSize);
            }

            rowPointers.Add(values.Count);
        }

        return new SparseMatrixCSR(n, n, values.ToArray(), colIndices.ToArray(), rowPointers.ToArray());
    }

    // Create a simple right-hand side vector
    static double[] CreateRightHandSide(int n)
    {
        double[] b = new double[n];
        for (int i = 0; i < n; i++)
        {
            // Simple test: b[i] = 1.0 (or use a more complex function)
            b[i] = 1.0;
        }
        return b;
    }
}