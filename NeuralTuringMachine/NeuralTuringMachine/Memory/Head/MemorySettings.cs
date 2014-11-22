namespace NeuralTuringMachine.Memory.Head
{
    public class MemorySettings
    {
        public int MemoryCellCount { get; private set; }
        public int MemoryVectorLength { get; private set; }
        public int MaxConvolutionalShift { get; private set; }
        public int ReadHeadCount { get; private set; }
        public int WriteHeadCount { get; private set; }
        public int ReadHeadLength
        {
            get { return MemoryVectorLength + AddressingNeuronCount; }
        }
        public int WriteHeadLength
        {
            get { return (MemoryVectorLength*3) + AddressingNeuronCount; }
        }
        public int AddressingNeuronCount
        {
            get
            {
                return 4 + (MaxConvolutionalShift * 2);
            }
        }

        public MemorySettings(int memoryCellCount, int memoryVectorLength, int maxConvolutionalShift, int readHeadCount, int writeHeadCount)
        {
            MemoryCellCount = memoryCellCount;
            MemoryVectorLength = memoryVectorLength;
            MaxConvolutionalShift = maxConvolutionalShift;
            ReadHeadCount = readHeadCount;
            WriteHeadCount = writeHeadCount;
        }
    }
}
