using System;
using AForge;
using AForge.Genetic;
using AForge.Math.Random;
using AForge.Neuro.Learning;
using NeuralTuringMachine.Controller;
using NeuralTuringMachine.GeneticsOptimalization;

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

            double[][] _idealOutputs = new double[inputCount][];

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

                ControllerInput input = _bpttMachines[i].GetInputForController(inputs[i], i > 0 ? _bpttMachines[i - 1].LastControllerOutput : null);

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
                    double[] idealInput = FindIdealInput(inputs[i + 1], _idealOutputs[i + 1], _bpttMachines[i + 1]);
                    double[] readHeadIdealOutput = FindReadHeadIdealOutput();
                    double[] writeHeadIdealOutput = FindWriteHeadIdealOutput();
                    //FIND OUT WHAT SHOULD BE OUTPUT OF READ HEAD       - genetic
                    //FIND OUT WHAT SHOULD BE OUTPUT OF WRITE HEAD      - genetic
                }

                _idealOutputs[i] = output;

                bpTeacher.Run(input.Input, output);
            }

            //Average the networks


        }

        private double[] FindWriteHeadIdealOutput()
        {
            return null;
        }

        private double[] FindReadHeadIdealOutput()
        {
            return null;
        }

        public double[] FindIdealInput(double[] input, double[] idealOutput, NeuralTuringMachine ntm)
        {
            int chromosomeLength = ntm.Controller.InputsCount - ntm.InputCount;
            Population population =
                new Population(
                    100,
                    new ControllerInputChromosome(new UniformGenerator(new Range(0, 1)), new UniformGenerator(new Range(0, 1)), new UniformGenerator(new Range(0, 1)), chromosomeLength), 
                    new IdealInputFitnessFunction(input, idealOutput, ntm.Controller),
                    new RouletteWheelSelection());

            for (int i = 0; i < 1000; i++)
            {
                population.RunEpoch();
            }

            ControllerInputChromosome bestChromosome = (ControllerInputChromosome)population.BestChromosome;
            return bestChromosome.Value;
        }
    }
}
