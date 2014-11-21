using AForge.Genetic;
using AForge.Math.Metrics;
using NeuralTuringMachine.Memory;

namespace NeuralTuringMachine.GeneticsOptimalization
{
    class IdealMemoryContentFitnessFunction : IFitnessFunction
    {
        private readonly double[] _idealReadData;
        private readonly double[] _readWeights;
        private readonly NtmMemory _memory;
        private readonly IDistance _distance;

        public IdealMemoryContentFitnessFunction(double[] idealReadData, double[] readWeights, int memoryCellCount, int memoryVectorLength)
        {
            _idealReadData = idealReadData;
            _readWeights = readWeights;
            _memory = new NtmMemory(memoryCellCount, memoryVectorLength);
            _distance = new EuclideanDistance();
        }

        public double Evaluate(IChromosome chromosome)
        {
            DoubleArrayChromosome doubleArrayChromosome = (DoubleArrayChromosome)chromosome;
            double[] memoryContent = doubleArrayChromosome.Value;

            _memory.SetMemoryContent(memoryContent);
            double[] readData = _memory.Read(_readWeights);

            double distance = _distance.GetDistance(readData, _idealReadData);
            return 1 / (1 + distance);
        }
    }
}
