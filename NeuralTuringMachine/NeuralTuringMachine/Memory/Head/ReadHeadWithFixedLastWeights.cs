namespace NeuralTuringMachine.Memory.Head
{
    public class ReadHeadWithFixedLastWeights : ReadHead
    {
        public ReadHeadWithFixedLastWeights(double[] lastWeights, MemorySettings settings) : base(settings)
        {
            LastWeights = lastWeights;
        }
    }
}
