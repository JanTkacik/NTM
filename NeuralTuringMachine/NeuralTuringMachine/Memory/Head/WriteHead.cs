using System;
using NeuralTuringMachine.Misc;

namespace NeuralTuringMachine.Memory.Head
{
    public class WriteHead : Head
    {
        private readonly int _outputNeuronCount;

        public double[] EraseVector { get; private set; }
        public double[] AddVector { get; private set; }

        public override int OutputNeuronCount { get { return _outputNeuronCount; } }

        public WriteHead(int memoryLength, int memoryCellSize, int maxConvShift) : base(memoryLength, memoryCellSize, maxConvShift)
        {
            //ADD, ERASE, KEY vectors
            _outputNeuronCount = (MemoryCellSize * 3) + AddressingNeuronsCount;

            EraseVector = new double[MemoryCellSize];
            AddVector = new double[MemoryCellSize];
        }

        private WriteHead(
            double[] eraseVector, 
            double[] addVector, 
            double[] lastWeights, 
            AddressingData addressingData, 
            int memoryLength, int memoryCellSize, int maxConvolutialShift) : base(memoryLength, memoryCellSize, maxConvolutialShift)
        {
            _outputNeuronCount = (MemoryCellSize * 3) + AddressingNeuronsCount;
            EraseVector = eraseVector;
            AddVector = addVector;
            LastWeights = lastWeights;
            ActualAddressingData = addressingData;
        }

        public void UpdateEraseVector(double[] writeHeadOutput)
        {
            Array.Copy(writeHeadOutput, MemoryCellSize + AddressingNeuronsCount, EraseVector, 0, MemoryCellSize);
        }

        public void UpdateAddVector(double[] writeHeadOutput)
        {
            Array.Copy(writeHeadOutput, (MemoryCellSize * 2) + AddressingNeuronsCount, AddVector, 0, MemoryCellSize);
        }

        public void UpdateMemory(NtmMemory memory)
        {
            double[] weightVector = GetWeightVector(memory);
            LastWeights = weightVector;
            memory.Write(weightVector, EraseVector, AddVector);
        }

        public WriteHead Clone()
        {
            if (ActualAddressingData != null)
            {
                return new WriteHead(
                    ArrayHelper.CloneArray(EraseVector),
                    ArrayHelper.CloneArray(AddVector),
                    ArrayHelper.CloneArray(LastWeights),
                    ActualAddressingData.Clone(),
                    MemoryLength, MemoryCellSize, MaxConvolutialShift);
            }
            return new WriteHead(
                    ArrayHelper.CloneArray(EraseVector),
                    ArrayHelper.CloneArray(AddVector),
                    ArrayHelper.CloneArray(LastWeights),
                    null,
                    MemoryLength, MemoryCellSize, MaxConvolutialShift);
        }
    }
}
