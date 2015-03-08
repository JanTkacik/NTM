using System;

namespace NTM2.Controller
{
    internal class OutputLayer
    {
        private readonly UnitFactory _unitFactory;

        //Weights from controller to output
        internal readonly Unit[][] _wyh1;

        //Weights from controller to head
        internal readonly Unit[][][] _wuh1;

        public OutputLayer(int outputSize, int controllerSize, int headCount, int headUnitSize, UnitFactory unitFactory)
        {
            _unitFactory = unitFactory;
            _wyh1 = _unitFactory.GetTensor2(outputSize, controllerSize + 1);
            _wuh1 = _unitFactory.GetTensor3(headCount, headUnitSize, controllerSize + 1);
        }

        private OutputLayer(Unit[][] wyh1, Unit[][][] wuh1, UnitFactory unitFactory)
        {
            _wyh1 = wyh1;
            _wuh1 = wuh1;
            _unitFactory = unitFactory;
        }

        public void ForwardPropagation()
        {
            
        }

        public OutputLayer Clone()
        {
            return new OutputLayer(_wyh1, _wuh1, _unitFactory);
        }

        public void BackwardErrorPropagation()
        {
            
        }

        public void UpdateWeights(Action<Unit> updateAction)
        {
            
        }
    }
}
