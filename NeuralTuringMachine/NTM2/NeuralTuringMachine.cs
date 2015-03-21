using NTM2.Controller;
using NTM2.Learning;
using NTM2.Memory;

namespace NTM2
{
    //TODO add saving and loading from file
    public sealed class NeuralTuringMachine
    {
        #region Fiels
        private readonly FeedForwardController _controller;
        private readonly NTMMemory _memory;

        private MemoryState _oldMemoryState;
        private MemoryState _newMemoryState;

        private double[] _lastInput; 
        #endregion
        
        #region Ctors

        internal NeuralTuringMachine(NeuralTuringMachine oldMachine)
        {
            _controller = oldMachine._controller.Clone();
            _memory = oldMachine._memory;
            _newMemoryState = oldMachine._newMemoryState;
            _oldMemoryState = oldMachine._oldMemoryState;
        }

        public NeuralTuringMachine(int inputSize, int outputSize, int controllerSize, int headCount, int memoryColumnsN, int memoryRowsM, IWeightUpdater initializer)
        {
            _memory = new NTMMemory(memoryColumnsN, memoryRowsM, headCount);
            _controller = new FeedForwardController(controllerSize, inputSize, outputSize, headCount, memoryRowsM);
            UpdateWeights(initializer);
        }

        #endregion

        #region Public Methods

        public void Process(double[] input)
        {
            _lastInput = input;
            _oldMemoryState = _newMemoryState;

            _controller.ForwardPropagation(input, _oldMemoryState);
            _newMemoryState = new MemoryState(_controller.OutputLayer.HeadsNeurons, _oldMemoryState.Memory);
        }

        public double[] GetOutput()
        {
            return _controller.GetOutput();
        } 

        #endregion
        
        #region Internal Methods

        internal void InitializeMemoryState()
        {
            _newMemoryState = new MemoryState(_memory);
            _newMemoryState.DoInitialReading();
            _oldMemoryState = null;
        }

        internal void BackwardErrorPropagation(double[] knownOutput)
        {
            _newMemoryState.BackwardErrorPropagation();
            _controller.BackwardErrorPropagation(knownOutput, _lastInput, _oldMemoryState.ReadData);
        }

        internal void BackwardErrorPropagation()
        {
            _newMemoryState.BackwardErrorPropagation2();
        }

        internal void UpdateWeights(IWeightUpdater weightUpdater)
        {
            _memory.UpdateWeights(weightUpdater);
            _controller.UpdateWeights(weightUpdater);
        } 

        #endregion
    }
}
