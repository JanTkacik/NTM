namespace NeuralTuringMachine.Memory.Head
{
    public class ReadHeadWithFixedLastWeights : ReadHead
    {
        public ReadHeadWithFixedLastWeights(double[] lastWeights, int memoryLength, int memoryCellSize, int maxConvolutialShift) : base(memoryLength, memoryCellSize, maxConvolutialShift)
        {
            LastWeights = lastWeights;
        }
    }
}
