using System;
using NTM2.Controller;

namespace NTM2.Learning
{
    //SEE http://arxiv.org/pdf/1308.0850v5.pdf page 23
    public class RMSPropWeightUpdater : WeightUpdaterBase
    {
        public double EwmaDecayRate { get; private set; }
        public double Momentum { get; private set; }
        public double LearningRate { get; private set; }
        public double RMSRegularizer { get; private set; }
        private readonly double[] _n;
        private readonly double[] _g;
        private readonly double[] _delta;
        private int _i;

        //EWMA - Exponentially weighted moving average
        public RMSPropWeightUpdater(int weightsCount, double ewmaDecayRate = 0.95, double momentum = 0.5, double learningRate = 0.0001, double rmsRegularizer = 0.001)
        {
            EwmaDecayRate = ewmaDecayRate;
            Momentum = momentum;
            LearningRate = learningRate;
            RMSRegularizer = rmsRegularizer;
            _n = new double[weightsCount];
            _g = new double[weightsCount];
            _delta = new double[weightsCount];
            _i = 0;
        }

        public override void Reset()
        {
            _i = 0;
        }

        public override void UpdateWeight(Unit unit)
        {
            _n[_i] = (EwmaDecayRate * _n[_i]) + ((1 - EwmaDecayRate) * unit.Gradient * unit.Gradient);
            _g[_i] = (EwmaDecayRate * _g[_i]) + ((1 - EwmaDecayRate) * unit.Gradient);
            _delta[_i] = (Momentum * _delta[_i]) - (LearningRate * (unit.Gradient / Math.Sqrt(_n[_i] - (_g[_i] * _g[_i]) + RMSRegularizer)));
            unit.Value += _delta[_i];
            _i++;
        }

        public RMSPropWeightUpdater Clone()
        {
            RMSPropWeightUpdater copy = new RMSPropWeightUpdater(_n.Length, EwmaDecayRate, Momentum, LearningRate, RMSRegularizer);
            Array.Copy(_n, copy._n, _n.Length);
            Array.Copy(_g, copy._g, _g.Length);
            Array.Copy(_delta, copy._delta, _delta.Length);
            return copy;
        }
    }
}
