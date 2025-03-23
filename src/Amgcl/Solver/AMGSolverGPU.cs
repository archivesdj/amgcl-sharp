using Amgcl.Matrix;
using Amgcl.Enums;
using Amgcl.Helpers;
using Amgcl.Coarsening;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda; // Assuming CUDA GPU

namespace Amgcl.Solver;

public class AMGSolverGPU : ISolver
{
    private readonly Accelerator _accelerator;
    private readonly List<IAMGLevel> _levels;
    private readonly SparseMatrixCSR _A;
    private readonly List<ISolver> _smoothers; // Still CPU-based smoothers for simplicity
    private ISolver? _coarseSolver;

    // Configuration properties
    public CoarseningType CoarseningType { get; set; } = CoarseningType.SmoothedAggregation;
    public int MaxLevels { get; set; } = 10;
    public SolverType SmootherType { get; set; } = SolverType.CGGPU;
     public int PreSmootherIterations { get; set; } = 2;
    public int PostSmootherIterations { get; set; } = 2;
    public int MinGridSize { get; set; } = 8;

    // Constructor: Initialize with matrix and GPU context
    public AMGSolverGPU(SparseMatrixCSR A, Accelerator? accelerator)
    {
        this._A = A ?? throw new ArgumentNullException(nameof(A));
        _levels = new List<IAMGLevel>();
        _smoothers = new List<ISolver>();

        _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
         // deviceType will be "GPU (CUDA)" or "CPU"
        string deviceType = _accelerator.AcceleratorType switch
        {
            AcceleratorType.Cuda => "GPU (CUDA)",
            AcceleratorType.CPU => "CPU",
            _ => "Unknown"
        };
        Console.WriteLine($"Using {deviceType} for computation (Name: {_accelerator.Name}, MaxThreads: {_accelerator.MaxNumThreads})");
        Console.WriteLine($"Maximum multi-level set to: {MaxLevels}");
    }

    // Build the multigrid hierarchy (unchanged from CPU version)
    public void BuildHierarchy()
    {
        _levels.Clear();
        _smoothers.Clear();

        IAMGLevel fineLevel = AMGLevelFactory.CreateLevel(CoarseningType, _A, MinGridSize);
        _levels.Add(fineLevel);
        _smoothers.Add(SolverFactory.CreateSolver(SmootherType, fineLevel.A, _accelerator));

        IAMGLevel current = _levels[0];
        while (_levels.Count < MaxLevels)
        {
            IAMGLevel? coarse = current.Coarsen();
            if (coarse == null) break;
            _levels.Add(coarse);
            _smoothers.Add(SolverFactory.CreateSolver(SmootherType, coarse.A, _accelerator));
            current = coarse;
        }

        if (_levels.Count > 0)
        {
            _smoothers.RemoveAt(_smoothers.Count - 1); // Remove smoother for the coarsest level
            _coarseSolver = SolverFactory.CreateSolver(SolverType.LUDirectGPU, _levels[_levels.Count - 1].A, _accelerator);
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

    // V-Cycle with GPU acceleration
    private void VCycle(int level, double[] b, double[] x, int preSmooth, int postSmooth, double tolerance)
    {
        if (level == _levels.Count - 1) // Coarsest level
        {
            _coarseSolver!.Relax(b, x, 1, tolerance); // Direct solve on CPU
            return;
        }

        IAMGLevel current = _levels[level];
        ISolver currentSmoother = _smoothers[level];

        // Pre-smoothing (CPU-based for now)
        currentSmoother.Relax(b, x, preSmooth, tolerance);

        // GPU buffers for current level
        using var bBuffer = _accelerator.Allocate1D<double>(b);
        using var xBuffer = _accelerator.Allocate1D<double>(x);
        using var residualBuffer = _accelerator.Allocate1D<double>(current.Residual);
        using var aValuesBuffer = _accelerator.Allocate1D<double>(current.A.Values);
        using var aColIndicesBuffer = _accelerator.Allocate1D<int>(current.A.ColIndices);
        using var aRowPointersBuffer = _accelerator.Allocate1D<int>(current.A.RowPointers);

        // Load and execute residual kernel
        var residualKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<int, Stride1D.Dense>, 
            ArrayView1D<int, Stride1D.Dense>>(ResidualKernel);
        residualKernel(
            current.A.Rows, 
            bBuffer.View, 
            xBuffer.View, 
            residualBuffer.View, 
            aValuesBuffer.View, 
            aColIndicesBuffer.View, 
            aRowPointersBuffer.View);
        _accelerator.Synchronize();
        residualBuffer.CopyToCPU(current.Residual);

        // GPU buffers for coarse level restriction
        IAMGLevel coarse = _levels[level + 1];
        using var coarseResidualBuffer = _accelerator.Allocate1D<double>(coarse.Residual);
        using var rValuesBuffer = _accelerator.Allocate1D<double>(current.R!.Values);
        using var rColIndicesBuffer = _accelerator.Allocate1D<int>(current.R!.ColIndices);
        using var rRowPointersBuffer = _accelerator.Allocate1D<int>(current.R!.RowPointers);

        // Load and execute restriction kernel
        var restrictionKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<int, Stride1D.Dense>, 
            ArrayView1D<int, Stride1D.Dense>>(RestrictionKernel);
        restrictionKernel(
            current.R!.Rows, 
            residualBuffer.View, 
            coarseResidualBuffer.View, 
            rValuesBuffer.View, 
            rColIndicesBuffer.View, 
            rRowPointersBuffer.View);
        _accelerator.Synchronize();
        coarseResidualBuffer.CopyToCPU(coarse.Residual);

        // Recursive V-Cycle
        Array.Clear(coarse.Correction, 0, coarse.Correction.Length);
        VCycle(level + 1, coarse.Residual, coarse.Correction, preSmooth, postSmooth, tolerance);

        // GPU buffers for prolongation
        using var coarseCorrectionBuffer = _accelerator.Allocate1D<double>(coarse.Correction);
        coarseCorrectionBuffer.CopyFromCPU(coarse.Correction);
        using var fineCorrectionBuffer = _accelerator.Allocate1D<double>(current.Correction);
        using var pValuesBuffer = _accelerator.Allocate1D<double>(current.P!.Values);
        using var pColIndicesBuffer = _accelerator.Allocate1D<int>(current.P!.ColIndices);
        using var pRowPointersBuffer = _accelerator.Allocate1D<int>(current.P!.RowPointers);

        // Load and execute prolongation kernel
        var prolongationKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<int, Stride1D.Dense>, 
            ArrayView1D<int, Stride1D.Dense>>(ProlongationKernel);
        prolongationKernel(
            current.P!.Rows, 
            coarseCorrectionBuffer.View, 
            fineCorrectionBuffer.View, 
            pValuesBuffer.View, 
            pColIndicesBuffer.View, 
            pRowPointersBuffer.View);
        _accelerator.Synchronize();
        fineCorrectionBuffer.CopyToCPU(current.Correction);

        // GPU-accelerated solution update
        var updateKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>>(UpdateSolutionKernel);
        updateKernel(
            x.Length, 
            xBuffer.View, 
            fineCorrectionBuffer.View);
        _accelerator.Synchronize();
        xBuffer.CopyToCPU(x);

        // Post-smoothing (CPU-based for now)
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

            VCycle(0, b, x, PreSmootherIterations, PostSmootherIterations, tolerance); // 2 pre- and post-smoothing steps

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

    // GPU Kernels with SparseMatrixCSR components separated
    private static void ResidualKernel(
        Index1D index, 
        ArrayView1D<double, Stride1D.Dense> b, 
        ArrayView1D<double, Stride1D.Dense> x, 
        ArrayView1D<double, Stride1D.Dense> residual, 
        ArrayView1D<double, Stride1D.Dense> values, 
        ArrayView1D<int, Stride1D.Dense> colIndices, 
        ArrayView1D<int, Stride1D.Dense> rowPointers)
    {
        if (index >= rowPointers.Length - 1) return;

        double ax = 0.0;
        int start = rowPointers[index];
        int end = rowPointers[index + 1];
        for (int j = start; j < end; j++)
        {
            ax += values[j] * x[colIndices[j]];
        }
        residual[index] = b[index] - ax;
    }

    private static void RestrictionKernel(
        Index1D index, 
        ArrayView1D<double, Stride1D.Dense> fineResidual, 
        ArrayView1D<double, Stride1D.Dense> coarseResidual, 
        ArrayView1D<double, Stride1D.Dense> values, 
        ArrayView1D<int, Stride1D.Dense> colIndices, 
        ArrayView1D<int, Stride1D.Dense> rowPointers)
    {
        if (index >= rowPointers.Length - 1) return;

        double sum = 0.0;
        int start = rowPointers[index];
        int end = rowPointers[index + 1];
        for (int j = start; j < end; j++)
        {
            sum += values[j] * fineResidual[colIndices[j]];
        }
        coarseResidual[index] = sum;
    }

    private static void ProlongationKernel(
        Index1D index, 
        ArrayView1D<double, Stride1D.Dense> coarseCorrection, 
        ArrayView1D<double, Stride1D.Dense> fineCorrection, 
        ArrayView1D<double, Stride1D.Dense> values, 
        ArrayView1D<int, Stride1D.Dense> colIndices, 
        ArrayView1D<int, Stride1D.Dense> rowPointers)
    {
        if (index >= rowPointers.Length - 1) return;

        double sum = 0.0;
        int start = rowPointers[index];
        int end = rowPointers[index + 1];
        for (int j = start; j < end; j++)
        {
            sum += values[j] * coarseCorrection[colIndices[j]];
        }
        fineCorrection[index] = sum;
    }

    private static void UpdateSolutionKernel(
        Index1D index, 
        ArrayView1D<double, Stride1D.Dense> x, 
        ArrayView1D<double, Stride1D.Dense> correction)
    {
        if (index >= x.Length) return;
        x[index] += correction[index];
    }
}