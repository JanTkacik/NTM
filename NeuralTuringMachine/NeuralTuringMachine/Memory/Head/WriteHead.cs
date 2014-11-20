using System;

namespace NeuralTuringMachine.Memory.Head
{
    class WriteHead : Head
    {
        private readonly int _outputNeuronCount;

        private readonly double[] _eraseVector;
        private readonly double[] _addVector;

        public override int OutputNeuronCount { get { return _outputNeuronCount; } }

        public WriteHead(int memoryLength, int memoryCellSize, int id, int maxConvShift) : base(memoryLength, memoryCellSize, id, maxConvShift)
        {
            //ADD, ERASE, KEY vectors
            _outputNeuronCount = (MemoryCellSize * 3) + AddressingNeuronsCount;

            _eraseVector = new double[MemoryCellSize];
            _addVector = new double[MemoryCellSize];
        }

        public void UpdateEraseVector(double[] writeHeadOutput)
        {
            Array.Copy(writeHeadOutput, MemoryCellSize + AddressingNeuronsCount, _eraseVector, 0, MemoryCellSize);
        }

        public void UpdateAddVector(double[] writeHeadOutput)
        {
            Array.Copy(writeHeadOutput, (MemoryCellSize * 2) + AddressingNeuronsCount, _addVector, 0, MemoryCellSize);
        }

        public void UpdateMemory(NtmMemory memory)
        {
            double[] weightVector = GetWeightVector(memory);
            memory.Write(weightVector, _eraseVector, _addVector);
        }
    }
}
