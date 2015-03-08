using System;
using NTM2.Memory;
using NTM2.Memory.Addressing;

namespace NTM2.Controller
{
    //TODO refactor extract layers - input, hidden and output
    class FeedForwardController : IController
    {
        #region Fields and variables
        
        internal readonly HiddenLayer HiddenLayer;
        internal readonly OutputLayer OutputLayer;

        #endregion

        #region Ctor

        public FeedForwardController(int controllerSize, int inputSize, int outputSize, int headCount, int memoryUnitSizeM, UnitFactory unitFactory)
        {
            HiddenLayer = new HiddenLayer(controllerSize, inputSize, headCount, memoryUnitSizeM, unitFactory);
            OutputLayer = new OutputLayer(outputSize, controllerSize, headCount, memoryUnitSizeM, HiddenLayer, unitFactory);
        }

        private FeedForwardController(HiddenLayer hiddenLayer, OutputLayer outputLayer)
        {
            HiddenLayer = hiddenLayer;
            OutputLayer = outputLayer;
        }

        public IController Clone()
        {
            HiddenLayer newHiddenLayer = HiddenLayer.Clone();
            OutputLayer newOutputLayer = OutputLayer.Clone(newHiddenLayer);
            return new FeedForwardController(newHiddenLayer, newOutputLayer);
        }

        #endregion

        #region Forward propagation
        
        public void ForwardPropagation(double[] input, ReadData[] readData)
        {
            HiddenLayer.ForwardPropagation(input, readData);
            OutputLayer.ForwardPropagation();
        }

        #endregion

        #region Update weights

        public void UpdateWeights(Action<Unit> updateAction)
        {
            OutputLayer.UpdateWeights(updateAction);
            HiddenLayer.UpdateWeights(updateAction);
        }

        #endregion

        #region BackwardErrorPropagation

        public void BackwardErrorPropagation(double[] input, ReadData[] reads)
        {
            OutputLayer.BackwardErrorPropagation();
            HiddenLayer.BackwardErrorPropagation(input, reads);
        }

        #endregion
    }
}
