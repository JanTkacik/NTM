namespace NeuralTuringMachine.Memory
{
    class ReadHead : Head
    {
        private readonly int _outputNeuronCount;
        private readonly int _outputOffset;

        public override int OutputNeuronCount { get { return _outputNeuronCount; } }
        public override int OutputOffset { get { return _outputOffset; } }

        public ReadHead(int memoryLength, int memoryCellSize, int id, int readHeadOffset) : base(memoryLength, memoryCellSize, id)
        {
            _outputNeuronCount = MemoryCellSize + AddressingNeuronsCount;
            _outputOffset = readHeadOffset + (id * _outputNeuronCount);
        }

        public double[] GetVectorFromMemory(NTMMemory memory)
        {
            double[] weightVector = GetWeightVector(memory);
            return memory.Read(weightVector);
        }

        
    }
}
