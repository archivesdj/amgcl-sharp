using Amgcl.Matrix;
using Amgcl.Enums;
using Amgcl.Helpers;
using Amgcl.Coarsening;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda; // Assuming CUDA GPU

namespace Amgcl.Solver;

public class AMGSolverGPU : ISolver, IDisposable
{
    private readonly Context context;
    private readonly Accelerator accelerator;
    private List<IAMGLevel> levels;
    private SparseMatrixCSR matrix;
    private List<ISolver> smoothers; // Still CPU-based smoothers for simplicity
    private ISolver? coarseSolver;

    // Configuration properties
    public CoarseningType CoarseningType { get; set; } = CoarseningType.RugeStuben;
    public int MaxLevels { get; set; } = 5;
    public SolverType SmootherType { get; set; } = SolverType.GaussSeidel;
     public int PreSmootherIterations { get; set; } = 2;
    public int PostSmootherIterations { get; set; } = 2;
    public int MinGridSize { get; set; } = 4;

    // Constructor: Initialize with matrix and GPU context
    public AMGSolverGPU(SparseMatrixCSR matrix)
    {
        this.matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
        levels = new List<IAMGLevel>();
        smoothers = new List<ISolver>();

        // Initialize ILGPU context and accelerator
        context = Context.Create(builder => builder.Cuda()); // Use CUDA GPU
        accelerator = context.GetPreferredDevice(preferCPU: false)
                             .CreateAccelerator(context);

         // deviceType will be "GPU (CUDA)" or "CPU"
        string deviceType = accelerator.AcceleratorType switch
        {
            AcceleratorType.Cuda => "GPU (CUDA)",
            AcceleratorType.CPU => "CPU",
            _ => "Unknown"
        };
        Console.WriteLine($"Using {deviceType} for computation (Name: {accelerator.Name}, MaxThreads: {accelerator.MaxNumThreads})");
        Console.WriteLine($"Maximum multi-level set to: {MaxLevels}");
    }

    // Dispose method to clean up GPU resources
    public void Dispose()
    {
        accelerator.Dispose();
        context.Dispose();
    }

    // Build the multigrid hierarchy (unchanged from CPU version)
    public void BuildHierarchy()
    {
        levels.Clear();
        smoothers.Clear();

        IAMGLevel fineLevel = AMGLevelFactory.CreateLevel(CoarseningType, matrix, MinGridSize);
        levels.Add(fineLevel);
        smoothers.Add(SolverFactory.CreateSolver(SmootherType, fineLevel.A));

        IAMGLevel current = levels[0];
        while (levels.Count < MaxLevels)
        {
            IAMGLevel? coarse = current.Coarsen();
            if (coarse == null) break;
            levels.Add(coarse);
            smoothers.Add(SolverFactory.CreateSolver(SmootherType, coarse.A));
            current = coarse;
        }

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

    // V-Cycle with GPU acceleration
    private void VCycle(int level, double[] b, double[] x, int preSmooth, int postSmooth, double tolerance)
    {
        if (level == levels.Count - 1) // Coarsest level
        {
            coarseSolver!.Relax(b, x, 1, tolerance); // Direct solve on CPU
            return;
        }

        IAMGLevel current = levels[level];
        ISolver currentSmoother = smoothers[level];

        // Pre-smoothing (CPU-based for now)
        currentSmoother.Relax(b, x, preSmooth, tolerance);

        // GPU buffers for current level
        using var bBuffer = accelerator.Allocate1D<double>(b);
        using var xBuffer = accelerator.Allocate1D<double>(x);
        using var residualBuffer = accelerator.Allocate1D<double>(current.Residual);
        using var aValuesBuffer = accelerator.Allocate1D<double>(current.A.Values);
        using var aColIndicesBuffer = accelerator.Allocate1D<int>(current.A.ColIndices);
        using var aRowPointersBuffer = accelerator.Allocate1D<int>(current.A.RowPointers);

        // Load and execute residual kernel
        var residualKernel = accelerator.LoadAutoGroupedStreamKernel<
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
        accelerator.Synchronize();
        residualBuffer.CopyToCPU(current.Residual);

        // GPU buffers for coarse level restriction
        IAMGLevel coarse = levels[level + 1];
        using var coarseResidualBuffer = accelerator.Allocate1D<double>(coarse.Residual);
        using var rValuesBuffer = accelerator.Allocate1D<double>(current.R!.Values);
        using var rColIndicesBuffer = accelerator.Allocate1D<int>(current.R!.ColIndices);
        using var rRowPointersBuffer = accelerator.Allocate1D<int>(current.R!.RowPointers);

        // Load and execute restriction kernel
        var restrictionKernel = accelerator.LoadAutoGroupedStreamKernel<
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
        accelerator.Synchronize();
        coarseResidualBuffer.CopyToCPU(coarse.Residual);

        // Recursive V-Cycle
        Array.Clear(coarse.Correction, 0, coarse.Correction.Length);
        VCycle(level + 1, coarse.Residual, coarse.Correction, preSmooth, postSmooth, tolerance);

        // GPU buffers for prolongation
        using var coarseCorrectionBuffer = accelerator.Allocate1D<double>(coarse.Correction);
        coarseCorrectionBuffer.CopyFromCPU(coarse.Correction);
        using var fineCorrectionBuffer = accelerator.Allocate1D<double>(current.Correction);
        using var pValuesBuffer = accelerator.Allocate1D<double>(current.P!.Values);
        using var pColIndicesBuffer = accelerator.Allocate1D<int>(current.P!.ColIndices);
        using var pRowPointersBuffer = accelerator.Allocate1D<int>(current.P!.RowPointers);

        // Load and execute prolongation kernel
        var prolongationKernel = accelerator.LoadAutoGroupedStreamKernel<
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
        accelerator.Synchronize();
        fineCorrectionBuffer.CopyToCPU(current.Correction);

        // GPU-accelerated solution update
        var updateKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>>(UpdateSolutionKernel);
        updateKernel(
            x.Length, 
            xBuffer.View, 
            fineCorrectionBuffer.View);
        accelerator.Synchronize();
        xBuffer.CopyToCPU(x);

        // Post-smoothing (CPU-based for now)
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

    // Solve: Initialize x and call Relax
    public double[] Solve(double[] b, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows)
            throw new ArgumentException("Length of b does not match matrix rows.");

        double[] x = new double[matrix.Rows];
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