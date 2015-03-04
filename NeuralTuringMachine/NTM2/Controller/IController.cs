using System;

namespace NTM2.Controller
{
    interface IController
    {
        double ForwardPropagation(double sum, int neuronIndex, double[] input);
        void UpdateWeights(Action<Unit> updateAction);
        void BackwardErrorPropagation(double[] hiddenLayerGradients, double[] input);
    }
}
