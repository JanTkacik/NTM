namespace NeuralTuringMachine.Memory.Head
{
    class ReadHead : Head
    {
        private readonly int _outputNeuronCount;

        public override int OutputNeuronCount { get { return _outputNeuronCount; } }

        public ReadHead(int memoryLength, int memoryCellSize, int id, int maxConvolutialShift) : base(memoryLength, memoryCellSize, id, maxConvolutialShift)
        {
            _outputNeuronCount = MemoryCellSize + AddressingNeuronsCount;
        }

        public double[] GetVectorFromMemory(NtmMemory memory)
        {
            double[] weightVector = GetWeightVector(memory);
            return memory.Read(weightVector);
        }

        
    }
}
