using System;
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
        
        public NeuralTuringMachine(NeuralTuringMachine oldMachine, bool clone = true)
        {
            if (!clone)
            {
                Controller = oldMachine.Controller;
                Memory = oldMachine.Memory;

                _newMemoryState = new MemoryState(Memory);
                _newMemoryState.DoInitialReading();
                _oldMemoryState = null;
            }
            else
            {
                Controller = oldMachine.Controller.Clone();
                Memory = oldMachine.Memory;
                _newMemoryState = oldMachine._newMemoryState;
            }
        }

        public NeuralTuringMachine(int inputSize, int outputSize, int controllerSize, int headCount, int memoryColumnsN, int memoryRowsM)
        {
            Memory = new NTMMemory(memoryColumnsN, memoryRowsM, headCount);
            Controller = new FeedForwardController(controllerSize, inputSize, outputSize, headCount, memoryRowsM);
        }
        
        public void Process(double[] input)
        {
            _lastInput = input;
            _oldMemoryState = _newMemoryState;

            Controller.ForwardPropagation(input, _oldMemoryState.ReadData);

            for (int i = 0; i < Memory.HeadCount; i++)
            {
                Controller.OutputLayer.HeadsNeurons[i].OldHeadSettings = _oldMemoryState.HeadSettings[i];
            }
            
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

        //TODO replace with weight initialization in constructor
        public void UpdateWeights(Action<Unit> updateAction)
        {
            Memory.UpdateWeights(updateAction);
            Controller.UpdateWeights(updateAction);
        }

        public void UpdateWeights(IWeightUpdater weightUpdater)
        {
            Memory.UpdateWeights(weightUpdater);
            Controller.UpdateWeights(weightUpdater);
        }
    }
}
