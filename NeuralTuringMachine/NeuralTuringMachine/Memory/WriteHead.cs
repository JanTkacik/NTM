using System;

namespace NeuralTuringMachine.Memory
{
    class WriteHead : Head
    {
        private readonly int _outputNeuronCount;
        private readonly int _outputOffset;

        private readonly double[] _eraseVector;
        private readonly double[] _addVector;

        public override int OutputNeuronCount { get { return _outputNeuronCount; } }
        public override int OutputOffset { get { return _outputOffset; } }

        public WriteHead(int memoryLength, int memoryCellSize, int id, int writeHeadsOffset) : base(memoryLength, memoryCellSize, id)
        {
            //ADD, ERASE, KEY vectors
            _outputNeuronCount = (MemoryCellSize * 3) + AddressingNeuronsCount;
            _outputOffset = writeHeadsOffset + (_outputNeuronCount*id);

            _eraseVector = new double[MemoryCellSize];
            _addVector = new double[MemoryCellSize];
        }

        public void UpdateEraseVector(double[] output)
        {
            Array.Copy(output, _outputOffset + MemoryCellSize + AddressingNeuronsCount, _eraseVector, 0, MemoryCellSize);
        }

        public void UpdateAddVector(double[] output)
        {
            Array.Copy(output, _outputOffset + (MemoryCellSize * 2) + AddressingNeuronsCount, _addVector, 0, MemoryCellSize);
        }

        public void UpdateMemory(NTMMemory memory)
        {
            double[] weightVector = GetWeightVector(memory);
            memory.Write(weightVector, _eraseVector, _addVector);
        }
    }
}
