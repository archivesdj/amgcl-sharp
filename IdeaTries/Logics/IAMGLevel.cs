public interface IAMGLevel
{
    SparseMatrix A { get; }
    SparseMatrix? P { get; }
    SparseMatrix? R { get; }
    SparseMatrix? CoarseA { get; }
    int CoarseSize { get; }

    void BuildCoarseLevel(double strengthThreshold, double dampingFactor);
}