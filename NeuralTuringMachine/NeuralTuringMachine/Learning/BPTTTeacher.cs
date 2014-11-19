using System;
using System.Linq;
using AForge.Neuro.Learning;

namespace NeuralTuringMachine.Learning
{
    public class BpttTeacher
    {
        private readonly NeuralTuringMachine _originalMachine;
        private readonly double _learningRate;
        private readonly double _momentum;
        private NeuralTuringMachine[] _bpttMachines;
        private double[][] _ntmOutputs;
      
        public BpttTeacher(NeuralTuringMachine machine, double learningRate = 0.001, double momentum = 0)
        {
            _originalMachine = machine;
            _learningRate = learningRate;
            _momentum = momentum;
        }

        public void Run(double[][] inputs, double[][] outputs)
        {
            //Make copies of Neural turing machine for bptt
            int inputCount = inputs.Length;
            _bpttMachines = new NeuralTuringMachine[inputCount];
            _ntmOutputs = new double[inputCount][];
            for (int i = 0; i < inputCount; i++)
            {
                _bpttMachines[i] = _originalMachine.Clone();
            }

            //Forward propagation
            for (int i = 0; i < inputCount; i++)
            {
                if (i == 0)
                {
                    _ntmOutputs[i] = _bpttMachines[i].Compute(inputs[i]);
                }
                else
                {
                    _ntmOutputs[i] = _bpttMachines[i].Compute(inputs[i], _bpttMachines[i - 1]);
                }
            }

            double[] errors = new double[inputCount];

            //Backward error propagation
            for (int i = inputCount - 1; i >= 0; i--)
            {
                BackPropagationLearning bpTeacher = new BackPropagationLearning(_bpttMachines[i].Controller)
                    {
                        LearningRate = _learningRate,
                        Momentum = _momentum
                    };

                double[] input = _bpttMachines[i].GetInputForController(inputs[i], i > 0 ? _bpttMachines[i - 1].Controller.Output : null);

                double[] controllerOutput = _bpttMachines[i].Controller.Output;
                int outputLength = controllerOutput.Length;

                double[] output = new double[outputLength];

                int dataOutputLength = outputs[i].Length;
                Array.Copy(outputs[i], output, dataOutputLength);

                if (i == inputCount - 1)
                {
                    Array.Copy(controllerOutput, dataOutputLength, output, dataOutputLength, outputLength - dataOutputLength);
                }
                else
                {
                    //FIND OUT WHAT SHOULD BE READ
                    //FIND OUT WHAT SHOULD BE OUTPUT OF READ HEAD 
                    //FIND OUT WHAT SHOULD BE OUTPUT OF WRITE HEAD
                }

                bpTeacher.Run(input, output);
            }

            //Average the networks
            
            
        }
    }
}
