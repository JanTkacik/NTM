using AForge.Neuro.Learning;

namespace NeuralTuringMachine.Learning
{
    public class BpttTeacherWitKnownMemoryState
    {
        private readonly NeuralTuringMachine _originalMachine;
        private readonly double _learningRate;

        public BpttTeacherWitKnownMemoryState(NeuralTuringMachine machine, double learningRate = 0.01)
        {
            _originalMachine = machine;
            _learningRate = learningRate;
        }

        public void Run(double[][] inputs, double[][] outputs)
        {
            ResilientBackpropagationLearning bpTeacher = new ResilientBackpropagationLearning(_originalMachine.Controller)
                    {
                        LearningRate = _learningRate
                    };
            
            bpTeacher.RunEpoch(inputs, outputs);
        }
    }
}
