using NeuralTuringMachine.Misc;

namespace NeuralTuringMachine.Memory.Head
{
    public class ReadHead : Head
    {
        private readonly int _outputNeuronCount;

        public override int OutputNeuronCount { get { return _outputNeuronCount; } }

        public ReadHead(int memoryLength, int memoryCellSize, int maxConvolutialShift) : base(memoryLength, memoryCellSize, maxConvolutialShift)
        {
            _outputNeuronCount = MemoryCellSize + AddressingNeuronsCount;
        }

        private ReadHead(double[] lastWeights, AddressingData addressingData, int memoryLength, int memoryCellSize, int maxConvolutialShift)
            : base(memoryLength, memoryCellSize, maxConvolutialShift)
        {
            _outputNeuronCount = MemoryCellSize + AddressingNeuronsCount;
            LastWeights = lastWeights;
            ActualAddressingData = addressingData;
        }
        
        public double[] GetVectorFromMemory(NtmMemory memory)
        {
            double[] weightVector = GetWeightVector(memory);
            LastWeights = weightVector;
            return memory.Read(weightVector);
        }

        public ReadHead Clone()
        {
            if (ActualAddressingData != null)
            {
                return new ReadHead(ArrayHelper.CloneArray(LastWeights), ActualAddressingData.Clone(), MemoryLength, MemoryCellSize, MaxConvolutialShift);
            }
            return new ReadHead(ArrayHelper.CloneArray(LastWeights), null, MemoryLength, MemoryCellSize, MaxConvolutialShift);
        }
    }
}
