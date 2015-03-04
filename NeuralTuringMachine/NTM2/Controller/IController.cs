using System;
using NTM2.Memory;

namespace NTM2.Controller
{
    interface IController
    {
        int HiddenLayerSize { get; }
        double ForwardPropagation(double sum, int neuronIndex, double[] input, ReadData[] readData);
        void UpdateWeights(Action<Unit> updateAction);
        void BackwardErrorPropagation(double[] hiddenLayerGradients, double[] input, ReadData[] reads);
        IController Clone();
    }
}
