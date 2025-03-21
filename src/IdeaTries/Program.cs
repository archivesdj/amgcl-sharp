using AmgSharp.Tests;

namespace AmgSharp;

class Program
{
    static void Main(string[] args)
    {
        //SimpleTest.Run(usingMatrixMarket: false);

        PoissonTest.Run();

        //SimpleMatrixTest.Run();
        //Simple4MatrixTest.Run();
        //LaplaceTest.Run();

        //HeatEquationTest.Run();
    }
}