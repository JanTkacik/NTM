using System;
using NTM2.Memory.Addressing;

namespace NTM2.Controller
{
    class FeedForwardController : IController
    {
        private readonly UnitFactory _unitFactory;
        //Controller bias weights
        private readonly Unit[] _wh1b;
        
        private int ControllerSize { get { return _wh1b.Length; } }

        public FeedForwardController(Unit[] wh1b)
        {
            _wh1b = wh1b;
        }

        public FeedForwardController(int controllerSize, UnitFactory unitFactory)
        {
            _unitFactory = unitFactory;

            _wh1b = _unitFactory.GetVector(controllerSize);
        }

        public double ForwardPropagation(double tempSum, int i)
        {
            double sum = tempSum;
            sum += _wh1b[i].Value;
            return sum;
        }

        public void UpdateWeights(Action<Unit> updateAction)
        {
            Action<Unit[]> vectorUpdateAction = Unit.GetVectorUpdateAction(updateAction);
            vectorUpdateAction(_wh1b);
        }

        public void BackwardErrorPropagation(double[] hiddenGradients)
        {
            int controllerSize = ControllerSize;
            for (int i = 0; i < controllerSize; i++)
            {
                _wh1b[i].Gradient += hiddenGradients[i];
            }
        }
    }
}
