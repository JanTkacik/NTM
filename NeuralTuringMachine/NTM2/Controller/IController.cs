using System;

namespace NTM2.Controller
{
    interface IController
    {
        double ForwardPropagation();
        void UpdateWeights(Action<Unit> updateAction);
        void BackwardErrorPropagation(double[] hiddenGradients);
    }
}
