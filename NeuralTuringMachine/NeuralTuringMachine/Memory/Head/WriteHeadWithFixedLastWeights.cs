namespace NeuralTuringMachine.Memory.Head
{
    public class WriteHeadWithFixedLastWeights : WriteHead
    {
        public WriteHeadWithFixedLastWeights(double[] lastWeights, MemorySettings settings) : base(settings)
        {
            LastWeights = lastWeights;
        }
    }
}
