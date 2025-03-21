using Amgcl.Matrix;
using Amgcl.Solver;
using Amgcl.Helpers;
using Amgcl.Enums;

namespace AmgclXUnitTest.Solver;

public class SolverFactoryTests
{
    [Fact]
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
}