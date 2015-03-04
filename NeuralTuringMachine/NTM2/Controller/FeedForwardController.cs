using System;
using NTM2.Memory;

namespace NTM2.Controller
{
    //TODO refactor extract layers - input, hidden and output
    class FeedForwardController : IController
    {
        #region Fields and variables
        
        internal readonly HiddenLayer HiddenLayer;

        #endregion

        #region Ctor

        public FeedForwardController(int controllerSize, int inputSize, int headCount, int memoryUnitSizeM, UnitFactory unitFactory)
        {
            HiddenLayer = new HiddenLayer(controllerSize, inputSize, headCount, memoryUnitSizeM, unitFactory);
        }

        private FeedForwardController(HiddenLayer hiddenLayer)
        {
            HiddenLayer = hiddenLayer;
        }

        public IController Clone()
        {
            return new FeedForwardController(HiddenLayer.Clone());
        }

        #endregion

        #region Forward propagation
        
        public void ForwardPropagation(double[] input, ReadData[] readData)
        {
            HiddenLayer.ForwardPropagation(input, readData);
        }

        #endregion

        #region Update weights

        public void UpdateWeights(Action<Unit> updateAction)
        {
            HiddenLayer.UpdateWeights(updateAction);
        }

        #endregion

        #region BackwardErrorPropagation

        public void BackwardErrorPropagation(double[] input, ReadData[] reads)
        {
            HiddenLayer.BackwardErrorPropagation(input, reads);
        }

        #endregion
    }
}
