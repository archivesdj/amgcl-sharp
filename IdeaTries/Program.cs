
class Program
{
    static void Main(string[] args)
    {
        SimpleTest.Run(usingMatrixMarket: false);

        PoissonTest.Run();

        HeatEquationTest.Run();
    }
}