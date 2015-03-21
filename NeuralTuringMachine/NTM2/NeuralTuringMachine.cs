using NTM2.Controller;
using NTM2.Learning;
using NTM2.Memory;

namespace NTM2
{
    public class NeuralTuringMachine
    {
        internal readonly FeedForwardController Controller;
        internal readonly NTMMemory Memory;

        private MemoryState _oldMemoryState;
        private MemoryState _newMemoryState;

        private double[] _lastInput;

        public NeuralTuringMachine(NeuralTuringMachine oldMachine)
        {
            Controller = oldMachine.Controller.Clone();
            Memory = oldMachine.Memory;
            _newMemoryState = oldMachine._newMemoryState;
            _oldMemoryState = oldMachine._oldMemoryState;
        }

        public NeuralTuringMachine(int inputSize, int outputSize, int controllerSize, int headCount, int memoryColumnsN, int memoryRowsM, IWeightUpdater initializer)
        {
            Memory = new NTMMemory(memoryColumnsN, memoryRowsM, headCount);
            Controller = new FeedForwardController(controllerSize, inputSize, outputSize, headCount, memoryRowsM);
            UpdateWeights(initializer);
        }

        internal void InitializeMemoryState()
        {
            _newMemoryState = new MemoryState(Memory);
            _newMemoryState.DoInitialReading();
            _oldMemoryState = null;
        }

        public void Process(double[] input)
        {
            _lastInput = input;
            _oldMemoryState = _newMemoryState;

            Controller.ForwardPropagation(input, _oldMemoryState);
            _newMemoryState = new MemoryState(Controller.OutputLayer.HeadsNeurons, _oldMemoryState.Memory);
        }

        public void BackwardErrorPropagation(double[] knownOutput)
        {
            _newMemoryState.BackwardErrorPropagation();
            Controller.BackwardErrorPropagation(knownOutput, _lastInput, _oldMemoryState.ReadData);
        }

        public void BackwardErrorPropagation()
        {
            _newMemoryState.BackwardErrorPropagation2();
        }

        public double[] GetOutput()
        {
            return Controller.GetOutput();
        }

        public void UpdateWeights(IWeightUpdater weightUpdater)
        {
            Memory.UpdateWeights(weightUpdater);
            Controller.UpdateWeights(weightUpdater);
        }
    }
}
