using AForge.Genetic;
using AForge.Math.Metrics;
using NeuralTuringMachine.Memory;
using NeuralTuringMachine.Memory.Head;

namespace NeuralTuringMachine.GeneticsOptimalization
{
    class IdealWriteHeadOutputFitnessFunction : IFitnessFunction
    {
        private readonly double[] _idealMemoryContent;
        private readonly NtmMemory _currentMemory;
        private readonly IDistance _distance;
        private readonly WriteHeadWithFixedLastWeights _writeHead;

        public IdealWriteHeadOutputFitnessFunction(double[] idealMemoryContent, double[] lastWeights, int maxConvShift, NtmMemory currentMemory)
        {
            _idealMemoryContent = idealMemoryContent;
            _currentMemory = currentMemory;
            _distance = new EuclideanDistance();
            _writeHead = new WriteHeadWithFixedLastWeights(lastWeights, currentMemory.CellCount, currentMemory.MemoryVectorLength, maxConvShift);
        }

        public double Evaluate(IChromosome chromosome)
        {
            DoubleArrayChromosome doubleArrayChromosome = (DoubleArrayChromosome)chromosome;
            double[] addressingData = doubleArrayChromosome.Value;
            
            _writeHead.UpdateAddressingData(addressingData);
            _writeHead.UpdateAddVector(addressingData);
            _writeHead.UpdateEraseVector(addressingData);

            double[] weightVector = _writeHead.GetWeightVector(_currentMemory);

            double[] memoryContentAfter = _currentMemory.GetDataAfterWrite(weightVector, _writeHead.EraseVector, _writeHead.AddVector);

            return _distance.GetDistance(_idealMemoryContent, memoryContentAfter);
        }
    }
}
