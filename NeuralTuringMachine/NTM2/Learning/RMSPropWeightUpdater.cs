using System;
using NTM2.Controller;

namespace NTM2.Learning
{
    //SEE http://arxiv.org/pdf/1308.0850v5.pdf page 23
    public class RMSPropWeightUpdater : WeightUpdaterBase
    {
        public double GradientMomentum { get; private set; }
        public double DeltaMomentum { get; private set; }
        public double ChangeMultiplier { get; private set; }
        public double ChangeAddConstant { get; private set; }
        private readonly double[] _n;
        private readonly double[] _g;
        private readonly double[] _delta;
        private int _i;

        public RMSPropWeightUpdater(NeuralTuringMachine controller, double gradientMomentum = 0.95, double deltaMomentum = 0.5, double changeMultiplier = 0.001, double changeAddConstant = 0.001)
        {
            GradientMomentum = gradientMomentum;
            DeltaMomentum = deltaMomentum;
            ChangeMultiplier = changeMultiplier;
            ChangeAddConstant = changeAddConstant;
            _n = new double[controller.WeightsCount];
            _g = new double[controller.WeightsCount];
            _delta = new double[controller.WeightsCount];
            _i = 0;
        }

        public override void Reset()
        {
            _i = 0;
        }

        public override void UpdateWeight(Unit unit)
        {
            _n[_i] = (GradientMomentum * _n[_i]) + ((1 - GradientMomentum) * unit.Gradient * unit.Gradient);
            _g[_i] = (GradientMomentum * _g[_i]) + ((1 - GradientMomentum) * unit.Gradient);
            _delta[_i] = (DeltaMomentum * _delta[_i]) - (ChangeMultiplier * (unit.Gradient / Math.Sqrt(_n[_i] - (_g[_i] * _g[_i]) + ChangeAddConstant)));
            unit.Value += _delta[_i];
            _i++;
        }
    }
}
