using Amgcl.Coarsening;
using Amgcl.Enums;
using Amgcl.Matrix;

namespace Amgcl.Helpers;

public static class AMGLevelFactory
{
    // Create an AMG level based on the specified CoarseningType
    public static IAMGLevel CreateLevel(CoarseningType type, SparseMatrixCSR matrix, int minGridSize)
    {
        switch (type)
        {
            case CoarseningType.Vanilla:
                return new VanillaAMGLevel(matrix, minGridSize);
            case CoarseningType.RugeStuben:
                return new RugeStubenAMGLevel(matrix, minGridSize);
            case CoarseningType.Aggregation:
                return new AggregationAMGLevel(matrix, minGridSize);
            case CoarseningType.SmoothedAggregation:
                return new SmoothedAggregationAMGLevel(matrix, minGridSize);
            default:
                throw new ArgumentException("Unknown coarsening type specified.");
        }
    }
}