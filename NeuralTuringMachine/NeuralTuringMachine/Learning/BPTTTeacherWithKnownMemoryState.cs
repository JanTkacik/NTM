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
            _bpTeacher.RunEpoch(inputs, outputs);
        }
    }
}
