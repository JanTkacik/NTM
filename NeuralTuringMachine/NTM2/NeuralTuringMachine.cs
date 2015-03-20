using System;
using NTM2.Controller;
using NTM2.Learning;
using NTM2.Memory;
using NTM2.Memory.Addressing;

namespace NTM2
{
    public class NeuralTuringMachine
    {
        internal readonly NTMMemory Memory;
        internal readonly FeedForwardController Controller;
        
        public int HeadCount
        {
            get { return Controller.OutputLayer.HeadsNeurons.Length; }
        }

        public Head[] HeadsNeurons
        {
            get { return Controller.OutputLayer.HeadsNeurons; }
        }
        
        public NeuralTuringMachine(int inputSize, int outputSize, int controllerSize, int headCount, int memoryColumnsN, int memoryRowsM)
        {
            Memory = new NTMMemory(memoryColumnsN, memoryRowsM, headCount);
            
            Controller = new FeedForwardController(controllerSize, inputSize, outputSize, headCount, memoryRowsM);
            
        }

        private NeuralTuringMachine(FeedForwardController controller)
        {
            Controller = controller;
        }
        
        public NeuralTuringMachine Clone()
        {
            return new NeuralTuringMachine(Controller.Clone());
        }
        
        internal void ForwardPropagation(ReadData[] readData, double[] input)
        {
            Controller.ForwardPropagation(input, readData);
        }

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

        public double[] GetOutput()
        {
            return Controller.GetOutput();
        }
    }
}
