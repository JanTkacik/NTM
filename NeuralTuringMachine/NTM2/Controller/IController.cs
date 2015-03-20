using System;
using NTM2.Learning;
using NTM2.Memory;

namespace NTM2.Controller
{
    interface IController
    {
        void ForwardPropagation(double[] input, ReadData[] readData);
        //TODO remove
        void UpdateWeights(Action<Unit> updateAction);
        void UpdateWeights(IWeightUpdater weightUpdater);
        void BackwardErrorPropagation(double[] knownOutput, double[] input, ReadData[] reads);
        double[] GetOutput();
        IController Clone();
    }
}
