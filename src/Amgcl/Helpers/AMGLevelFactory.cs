using Amgcl.Coarsening;
using Amgcl.Enums;
using Amgcl.Matrix;

namespace Amgcl.Helpers;

public static class AMGLevelFactory
{
    // Create an AMG level based on the specified CoarseningType
    public static IAMGLevel CreateLevel(CoarseningType type, SparseMatrixCSR matrix)
    {
        switch (type)
        {
            case CoarseningType.Vanilla:
                return new VanillaAMGLevel(matrix);
            case CoarseningType.RugeStuben:
                return new RugeStubenAMGLevel(matrix);
            default:
                throw new ArgumentException("Unknown coarsening type specified.");
        }
    }
}