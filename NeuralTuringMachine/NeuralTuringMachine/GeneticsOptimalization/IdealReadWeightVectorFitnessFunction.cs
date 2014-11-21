using AForge.Genetic;
using AForge.Math.Metrics;
using NeuralTuringMachine.Memory;

namespace NeuralTuringMachine.GeneticsOptimalization
{
    public class IdealReadWeightVectorFitnessFunction : IFitnessFunction
    {
        private readonly double[] _idealReadValue;
        private readonly NtmMemory _memory;
        private readonly IDistance _distance; 

        public IdealReadWeightVectorFitnessFunction(double[] idealReadValue, NtmMemory memory)
        {
            _idealReadValue = idealReadValue;
            _memory = memory;
            _distance = new EuclideanDistance();
        }

        public double Evaluate(IChromosome chromosome)
        {
            DoubleArrayChromosome doubleArrayChromosome = (DoubleArrayChromosome)chromosome;
            double[] weightVector = doubleArrayChromosome.Value;

            double[] readFromMemory = _memory.Read(weightVector);

            return _distance.GetDistance(readFromMemory, _idealReadValue);
        }
    }
}
