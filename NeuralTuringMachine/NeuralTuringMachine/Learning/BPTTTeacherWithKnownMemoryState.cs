using System;
using AForge.Neuro.Learning;

namespace NeuralTuringMachine.Learning
{
    public class BpttTeacherWitKnownMemoryState
    {
        private readonly NTM _originalMachine;
        private readonly ResilientBackpropagationLearning _bpTeacher;

        public BpttTeacherWitKnownMemoryState(NTM machine)
        {
            _originalMachine = machine;
            _bpTeacher = new ResilientBackpropagationLearning(_originalMachine.Controller);
        }

        public void Run(double[][] inputs, double[][] outputs)
        {
            int notImprovementCounter = 0;
            double error = double.PositiveInfinity;
            while (notImprovementCounter < 100)
            {
                double errorNow = _bpTeacher.RunEpoch(inputs, outputs);
                double improvement = error - errorNow;

                if (improvement < 0.0000001)
                {
                    notImprovementCounter++;
                }
                else
                {
                    notImprovementCounter = 0;
                }

                error = errorNow;
            }
        }
    }
}
