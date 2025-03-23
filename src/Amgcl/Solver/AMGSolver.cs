using Amgcl.Matrix;
using Amgcl.Enums;
using Amgcl.Helpers;
using Amgcl.Coarsening;

namespace Amgcl.Solver;

public class AMGSolver : ISolver
{
    private readonly List<IAMGLevel> _levels;         // List of multigrid levels
    private readonly SparseMatrixCSR _A;         // Original matrix
    private readonly List<ISolver> _smoothers;        // List of smoothers for each level
    private ISolver? _coarseSolver;           // Direct solver for the coarsest level

    // Configuration properties
    public CoarseningType CoarseningType { get; set; } = CoarseningType.SmoothedAggregation;
    public int MaxLevels { get; set; } = 10;
    public SolverType SmootherType { get; set; } = SolverType.CG;
    public int PreSmootherIterations { get; set; } = 2;
    public int PostSmootherIterations { get; set; } = 2;
    public int MinGridSize { get; set; } = 8;

    // Constructor: Initialize with matrix only
    public AMGSolver(SparseMatrixCSR A)
    {
        this._A = A ?? throw new ArgumentNullException(nameof(A));
        _levels = new List<IAMGLevel>();
        _smoothers = new List<ISolver>();    // Initialize smoother list
    }

    // Build the multigrid hierarchy
    public void BuildHierarchy()
    {
        _levels.Clear();
        _smoothers.Clear();

        // Create the finest level
        IAMGLevel fineLevel = AMGLevelFactory.CreateLevel(CoarseningType, _A, MinGridSize);
        _levels.Add(fineLevel);
        _smoothers.Add(SolverFactory.CreateSolver(SmootherType, fineLevel.A));

        // Build hierarchy recursively
        IAMGLevel current = _levels[0];
        while (_levels.Count < MaxLevels)
        {
            var coarse = current.Coarsen();
            if (coarse == null) break;
            _levels.Add(coarse);
            _smoothers.Add(SolverFactory.CreateSolver(SmootherType, coarse.A));
            current = coarse;
        }

        // Initialize coarseSolver for the coarsest level
        if (_levels.Count > 0)
        {
            _smoothers.RemoveAt(_smoothers.Count - 1); // Remove smoother for the coarsest level
            _coarseSolver = SolverFactory.CreateSolver(SolverType.LUDirect, _levels[_levels.Count - 1].A);
        }
    
        // Print hierarchy information
        Console.WriteLine("===================================");
        Console.WriteLine($"Coarsening Method: {CoarseningType}");
        Console.WriteLine($"AMG hierarchy with {_levels.Count} levels:");
        for (int i = 0; i < _levels.Count; i++)
        {
            IAMGLevel level = _levels[i];
            Console.WriteLine($"Level {i}: {level.A.Rows}x{level.A.Cols}, {level.A.NonZeroCount} non-zeros");
        }
    }

    // V-Cycle: Recursive multigrid cycle
    private void VCycle(int level, double[] b, double[] x, int preSmooth, int postSmooth, double tolerance)
    {
        if (level == _levels.Count - 1) // Coarsest level
        {
            _coarseSolver!.Relax(b, x, 1, tolerance); // Direct solve
            return;
        }

        IAMGLevel current = _levels[level];
        ISolver currentSmoother = _smoothers[level]; // Use smoother for current level

        // Pre-smoothing
        currentSmoother.Relax(b, x, preSmooth, tolerance);

        // Compute residual: r = b - Ax
        current.A.Multiply(x, current.Residual);
        for (int i = 0; i < current.Residual.Length; i++)
            current.Residual[i] = b[i] - current.Residual[i];

        // Restrict residual to coarse level
        IAMGLevel coarse = _levels[level + 1];
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
        if (_coarseSolver == null || _levels.Count == 0)
            throw new InvalidOperationException("Solver hierarchy not built.");

        if (b.Length != _A.Rows || x.Length != _A.Rows)
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
        double[] x = new double[b.Length];
        Relax(b, x, maxIterations, tolerance);
        return x;
    }
}