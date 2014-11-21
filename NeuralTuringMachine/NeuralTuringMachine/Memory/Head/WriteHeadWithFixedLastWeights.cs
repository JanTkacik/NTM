namespace NeuralTuringMachine.Memory.Head
{
    public class WriteHeadWithFixedLastWeights : WriteHead
    {
        public WriteHeadWithFixedLastWeights(double[] lastWeights, int memoryLength, int memoryCellSize, int maxConvShift) : base(memoryLength, memoryCellSize, maxConvShift)
        {
            LastWeights = lastWeights;
        }
    }
}
