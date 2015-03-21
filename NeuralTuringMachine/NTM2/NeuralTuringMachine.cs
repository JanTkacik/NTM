using System;
using NTM2.Controller;
using NTM2.Learning;
using NTM2.Memory;

namespace NTM2
{
    public class NeuralTuringMachine
    {
        internal readonly FeedForwardController Controller;
        internal readonly MemoryState MemoryState;

        private readonly MemoryState _oldMemoryState;
        private MemoryState _memoryStateShit;

        private double[] _input;
        
        public NeuralTuringMachine(NeuralTuringMachine oldMachine, bool refactorShit = true)
        {
            if (!refactorShit)
            {
                Controller = oldMachine.Controller;
                MemoryState = oldMachine.MemoryState;

                _memoryStateShit = new MemoryState(MemoryState.Memory);
                _memoryStateShit.DoInitialReading();
                _oldMemoryState = null;
            }
            else
            {
                Controller = oldMachine.Controller.Clone();
                MemoryState = oldMachine.MemoryState;
                _oldMemoryState = oldMachine._memoryStateShit;
            }
        }

        public NeuralTuringMachine(int inputSize, int outputSize, int controllerSize, int headCount, int memoryColumnsN, int memoryRowsM)
        {
            MemoryState = new MemoryState(new NTMMemory(memoryColumnsN, memoryRowsM, headCount));
            MemoryState.DoInitialReading();
            Controller = new FeedForwardController(controllerSize, inputSize, outputSize, headCount, memoryRowsM);
        }
        
        public void ForwardPropagation(double[] input)
        {
            _input = input;

            Controller.ForwardPropagation(input, _oldMemoryState.ReadData);

            for (int i = 0; i < MemoryState.Memory.HeadCount; i++)
            {
                Controller.OutputLayer.HeadsNeurons[i].OldHeadSettings = _oldMemoryState.HeadSettings[i];
            }
            
            _memoryStateShit = new MemoryState(Controller.OutputLayer.HeadsNeurons, _oldMemoryState.Memory);
        }
        
        public void BackwardErrorPropagation(double[] knownOutput)
        {
            _memoryStateShit.BackwardErrorPropagation();
            Controller.BackwardErrorPropagation(knownOutput, _input, _oldMemoryState.ReadData);
        }

        public void BackwardErrorPropagation()
        {
            _memoryStateShit.BackwardErrorPropagation2();
        }

        public double[] GetOutput()
        {
            return Controller.GetOutput();
        }

        //TODO replace with weight initialization in constructor
        public void UpdateWeights(Action<Unit> updateAction)
        {
            MemoryState.Memory.UpdateWeights(updateAction);
            Controller.UpdateWeights(updateAction);
        }

        public void UpdateWeights(IWeightUpdater weightUpdater)
        {
            MemoryState.Memory.UpdateWeights(weightUpdater);
            Controller.UpdateWeights(weightUpdater);
        }
    }
}
