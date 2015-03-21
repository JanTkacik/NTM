namespace NTM2.Learning
{
    public class BPTTTeacher : INTMTeacher
    {
        private readonly NeuralTuringMachine _machine;
        private readonly IWeightUpdater _weightUpdater;
        private readonly IWeightUpdater _gradientResetter;

        public BPTTTeacher(NeuralTuringMachine machine, IWeightUpdater weightUpdater)
        {
            _machine = machine;
            _weightUpdater = weightUpdater;
            _gradientResetter = new GradientResetter();
        }

        public double[][] Train(double[][] input, double[][] knownOutput)
        {
            NeuralTuringMachine[] machines = new NeuralTuringMachine[input.Length];

            //FORWARD phase
            NeuralTuringMachine originalMachine = new NeuralTuringMachine(_machine, false);
            machines[0] = new NeuralTuringMachine(originalMachine);
            machines[0].Process(input[0]);
            for (int i = 1; i < input.Length; i++)
            {
                machines[i] = new NeuralTuringMachine(machines[i - 1]);
                machines[i].Process(input[i]);
            }

            //Gradient reset
            _gradientResetter.Reset();
            _machine.UpdateWeights(_gradientResetter);

            //BACKWARD phase
            for (int i = input.Length - 1; i >= 0; i--)
            {
                machines[i].BackwardErrorPropagation(knownOutput[i]);
            }
            originalMachine.BackwardErrorPropagation();

            //Weight updates
            _weightUpdater.Reset();
            _machine.UpdateWeights(_weightUpdater);
            
            return GetMachineOutputs(machines);
        }

        private double[][] GetMachineOutputs(NeuralTuringMachine[] machines)
        {
            double[][] realOutputs = new double[machines.Length][];
            for (int i = 0; i < machines.Length; i++)
            {
                NeuralTuringMachine machine = machines[i];
                realOutputs[i] = machine.GetOutput();
            }
            return realOutputs;
        }
    }
}
