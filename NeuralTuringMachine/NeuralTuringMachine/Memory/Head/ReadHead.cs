using NeuralTuringMachine.Misc;

namespace NeuralTuringMachine.Memory.Head
{
    public class ReadHead : Head
    {
        public ReadHead(MemorySettings memorySettings) : base(memorySettings)
        {
            
        }

        private ReadHead(double[] lastWeights, AddressingData addressingData, MemorySettings memorySettings)
            : base(memorySettings)
        {
            LastWeights = lastWeights;
            ActualAddressingData = addressingData;
        }

        public ReadHead(double[] rawControllerOutput, MemorySettings memorySettings) : base(memorySettings)
        {
            UpdateAddressingData(rawControllerOutput);
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
                return new ReadHead(ArrayHelper.CloneArray(LastWeights), ActualAddressingData.Clone(), MemorySettings);
            }
            return new ReadHead(ArrayHelper.CloneArray(LastWeights), null, MemorySettings);
        }

        public static double GetSimilarityScore(ReadHead headA, ReadHead headB)
        {
            return AddressingData.GetSimilarityScore(headA.ActualAddressingData, headB.ActualAddressingData);
        }
    }
}
