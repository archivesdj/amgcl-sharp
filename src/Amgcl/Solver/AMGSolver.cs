using Amgcl.Matrix;
using Amgcl.Enums;
using Amgcl.Helpers;
using Amgcl.Coarsening;

namespace Amgcl.Solver;

public class AMGSolver : ISolver
{
    private List<IAMGLevel> levels;         // List of multigrid levels
    private SparseMatrixCSR matrix;         // Original matrix
    private List<ISolver> smoothers;        // List of smoothers for each level
    private ISolver? coarseSolver;           // Direct solver for the coarsest level

    // Configuration properties
    public CoarseningType CoarseningType { get; set; } = CoarseningType.RugeStuben;
    public int MaxLevels { get; set; } = 5;
    public SolverType SmootherType { get; set; } = SolverType.CG;
    public int PreSmootherIterations { get; set; } = 2;
    public int PostSmootherIterations { get; set; } = 2;
    public int MinGridSize { get; set; } = 4;

    // Constructor: Initialize with matrix only
    public AMGSolver(SparseMatrixCSR matrix)
    {
        this.matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
        levels = new List<IAMGLevel>();
        smoothers = new List<ISolver>();    // Initialize smoother list
    }

    // Build the multigrid hierarchy
    public void BuildHierarchy()
    {
        levels.Clear();
        smoothers.Clear();

        // Create the finest level
        IAMGLevel fineLevel = AMGLevelFactory.CreateLevel(CoarseningType, matrix, MinGridSize);
        levels.Add(fineLevel);
        smoothers.Add(SolverFactory.CreateSolver(SmootherType, fineLevel.A));

        // Build hierarchy recursively
        IAMGLevel current = levels[0];
        while (levels.Count < MaxLevels)
        {
            var coarse = current.Coarsen();
            if (coarse == null) break;
            levels.Add(coarse);
            smoothers.Add(SolverFactory.CreateSolver(SmootherType, coarse.A));
            current = coarse;
        }

        // Initialize coarseSolver for the coarsest level
        if (levels.Count > 0)
        {
            smoothers.RemoveAt(smoothers.Count - 1); // Remove smoother for the coarsest level
            coarseSolver = SolverFactory.CreateSolver(SolverType.LUDirect, levels[levels.Count - 1].A);
        }
    
        // Print hierarchy information
        Console.WriteLine("===================================");
        Console.WriteLine($"Coarsening Method: {CoarseningType}");
        Console.WriteLine($"AMG hierarchy with {levels.Count} levels:");
        for (int i = 0; i < levels.Count; i++)
        {
            IAMGLevel level = levels[i];
            Console.WriteLine($"Level {i}: {level.A.Rows}x{level.A.Cols}, {level.A.NonZeroCount} non-zeros");
        }
    }

    // V-Cycle: Recursive multigrid cycle
    private void VCycle(int level, double[] b, double[] x, int preSmooth, int postSmooth, double tolerance)
    {
        if (level == levels.Count - 1) // Coarsest level
        {
            coarseSolver!.Relax(b, x, 1, tolerance); // Direct solve
            return;
        }

        IAMGLevel current = levels[level];
        ISolver currentSmoother = smoothers[level]; // Use smoother for current level

        // Pre-smoothing
        currentSmoother.Relax(b, x, preSmooth, tolerance);

        // Compute residual: r = b - Ax
        current.A.Multiply(x, current.Residual);
        for (int i = 0; i < current.Residual.Length; i++)
            current.Residual[i] = b[i] - current.Residual[i];

        // Restrict residual to coarse level
        IAMGLevel coarse = levels[level + 1];
        current.R!.Multiply(current.Residual, coarse.Residual);
       
        // Solve on coarse level recursively
        Array.Clear(coarse.Correction, 0, coarse.Correction.Length);
        VCycle(level + 1, coarse.Residual, coarse.Correction, preSmooth, postSmooth, tolerance);

        // Prolong correction to fine level
        current.P!.Multiply(coarse.Correction, current.Correction);

        // Update solution: x = x + correction
        for (int i = 0; i < x.Length; i++)
            x[i] += current.Correction[i];

        // Post-smoothing
        currentSmoother.Relax(b, x, postSmooth, tolerance);
    }

    // Relax: Perform AMG iterations (V-Cycles)
    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (coarseSolver == null || levels.Count == 0)
            throw new InvalidOperationException("Solver hierarchy not built.");

        if (b.Length != matrix.Rows || x.Length != matrix.Rows)
            throw new ArgumentException("Vector length does not match matrix dimensions.");

        for (int iter = 0; iter < maxIterations; iter++)
        {
            double[] prevX = new double[x.Length];
            Array.Copy(x, prevX, x.Length);

            // Perform one V-Cycle
            VCycle(0, b, x, PreSmootherIterations, PostSmootherIterations, tolerance);

            // Check convergence
            double maxDiff = 0;
            for (int i = 0; i < x.Length; i++)
            {
                double diff = Math.Abs(x[i] - prevX[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            if (maxDiff < tolerance) break;
        }
    }

    // Solve: Initialize x and call Relax to compute solution
    public double[] Solve(double[] b, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows)
            throw new ArgumentException("Length of b does not match matrix rows.");

        double[] x = new double[matrix.Rows];
        Relax(b, x, maxIterations, tolerance);
        return x;
    }
}