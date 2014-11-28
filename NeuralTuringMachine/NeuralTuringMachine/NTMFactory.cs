using NeuralTuringMachine.Memory.Head;

namespace NeuralTuringMachine
{
    public class NTMFactory
    {
        private int NTMId;

        public NTMFactory()
        {
            NTMId = 0;
        }

        public NTM CreateNTM(int inputCount, int outputCount, int hiddenNeuronsCount, int hiddenLayersCount, MemorySettings memorySettings)
        {
            NTM neuralTuringMachine = new NTM(inputCount, outputCount, hiddenNeuronsCount, hiddenLayersCount, memorySettings, NTMId);
            NTMId++;
            return neuralTuringMachine;
        }

        public NTM CloneNTM(NTM machine)
        {
            NTM neuralTuringMachine = machine.Clone(NTMId);
            NTMId++;
            return neuralTuringMachine;
        }
    }
}
