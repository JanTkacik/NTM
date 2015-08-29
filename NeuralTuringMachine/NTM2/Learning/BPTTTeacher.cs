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

        public double[][] TrainVerbose(double[][] input, double[][] knownOutput, out double[][] headAddressings)
        {
            var machines = TrainInternal(input, knownOutput);
            headAddressings = GetHeadAddressings(machines);
            return GetMachineOutputs(machines);
        }


        public double[][] Train(double[][] input, double[][] knownOutput)
        {
            var machines = TrainInternal(input, knownOutput);
            return GetMachineOutputs(machines);
        }

        public void TrainFast(double[][] input, double[][] knownOutput)
        {
            TrainInternal(input, knownOutput);
        }

        private NeuralTuringMachine[] TrainInternal(double[][] input, double[][] knownOutput)
        {
            NeuralTuringMachine[] machines = new NeuralTuringMachine[input.Length];

            //FORWARD phase
            _machine.InitializeMemoryState();
            machines[0] = new NeuralTuringMachine(_machine);
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
            _machine.BackwardErrorPropagation();

            //Weight updates
            _weightUpdater.Reset();
            _machine.UpdateWeights(_weightUpdater);
            return machines;
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

        private double[][] GetHeadAddressings(NeuralTuringMachine[] machines)
        {
            double[][] headAddressings = new double[machines.Length][];
            for (int i = 0; i < machines.Length; i++)
            {
                NeuralTuringMachine machine = machines[i];
                headAddressings[i] = machine.GetHeadAdressings();
            }
            return headAddressings;
        }
    }
}
