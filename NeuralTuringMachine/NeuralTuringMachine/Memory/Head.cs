namespace NeuralTuringMachine.Memory
{
    abstract class Head
    {
        public abstract int OutputNeuronCount { get; }
        public abstract int OutputOffset { get; }

        protected const int AddressingNeuronsCount = 4;
        protected int MemoryCellSize;
        protected AddressingData actualAddressingData;
        private readonly int _id;

        protected Head(int memoryCellSize, int id)
        {
            MemoryCellSize = memoryCellSize;
            _id = id;
        }

        public int ID
        {
            get { return _id; }
        }

        protected double[] GetWeightVector(NTMMemory memory)
        {
            throw new System.NotImplementedException();
        }

        public void UpdateAddressingData(double[] output)
        {
            actualAddressingData = new AddressingData(output, OutputOffset, OutputNeuronCount);
        }
    }
}
