using System;
using AForge.Math.Metrics;
using NeuralTuringMachine.Misc;

namespace NeuralTuringMachine.Memory.Head
{
    public class WriteHead : Head
    {
        private static ISimilarity _similarity = new EuclideanSimilarity();

        public double[] EraseVector { get; private set; }
        public double[] AddVector { get; private set; }

        public WriteHead(MemorySettings settings) : base(settings)
        {
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            //ADD, ERASE, KEY vectors
            EraseVector = new double[memoryVectorLength];
            AddVector = new double[memoryVectorLength];
        }

        private WriteHead(
            double[] eraseVector, 
            double[] addVector, 
            double[] lastWeights, 
            AddressingData addressingData,
            MemorySettings settings)
            : base(settings)
        {
            EraseVector = eraseVector;
            AddVector = addVector;
            LastWeights = lastWeights;
            ActualAddressingData = addressingData;
        }

        public WriteHead(double[] rawControllerOutput, MemorySettings settings) : this(settings)
        {
            UpdateEraseVector(rawControllerOutput);
            UpdateAddVector(rawControllerOutput);
            UpdateAddressingData(rawControllerOutput);
        }

        public void UpdateEraseVector(double[] writeHeadOutput)
        {
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            Array.Copy(writeHeadOutput, memoryVectorLength + MemorySettings.AddressingNeuronCount, EraseVector, 0, memoryVectorLength);
        }

        public void UpdateAddVector(double[] writeHeadOutput)
        {
            int memoryVectorLength = MemorySettings.MemoryVectorLength;
            Array.Copy(writeHeadOutput, (memoryVectorLength * 2) + MemorySettings.AddressingNeuronCount, AddVector, 0, memoryVectorLength);
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
                    MemorySettings);
            }
            return new WriteHead(
                    ArrayHelper.CloneArray(EraseVector),
                    ArrayHelper.CloneArray(AddVector),
                    ArrayHelper.CloneArray(LastWeights),
                    null,
                    MemorySettings);
        }

        public static double GetSimilarityScore(WriteHead headA, WriteHead headB)
        {
            double addressingSimilarityScore = AddressingData.GetSimilarityScore(headA.ActualAddressingData, headB.ActualAddressingData);
            double addSimilarityScore = _similarity.GetSimilarityScore(headA.AddVector, headB.AddVector);
            double eraseSimilarityScore = _similarity.GetSimilarityScore(headA.EraseVector, headB.EraseVector);

            return (addressingSimilarityScore + addSimilarityScore + eraseSimilarityScore)/3;
        }
    }
}
