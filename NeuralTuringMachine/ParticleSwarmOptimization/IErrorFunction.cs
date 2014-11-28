namespace ParticleSwarmOptimization
{
    public interface IErrorFunction
    {
        int Dimensions { get; }
        double MinForDimension(int dimension);
        double MaxForDimension(int dimension);
        double CalculateError(double[] position);
    }
}
