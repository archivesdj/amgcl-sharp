using Amgcl.Matrix;
using Amgcl.Solver;
using Amgcl.Enums;

namespace AmgclXUnitTest.Solver;

public class GPUSolverTests
{
    [Fact]
    public void Test_GPUSolve()
    {
        int gridSize = 10;
        int n = gridSize * gridSize;
        var matrix = Create2DPoissonMatrix(gridSize);
        double[] b = new double[n];
        for (int i = 0; i < n; i++) b[i] = 1.0;

        using var solver = new AMGSolverGPU(matrix)
        {
            CoarseningType = CoarseningType.SmoothedAggregation,
            MaxLevels = 4,
            SmootherType = SolverType.GaussSeidel
        };
        solver.BuildHierarchy();
        double[] x = solver.Solve(b, 100, 1e-6);

        Console.WriteLine("Solution (first 5): " + string.Join(" ", x.Take(5).Select(v => v.ToString("F4"))));
    }

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