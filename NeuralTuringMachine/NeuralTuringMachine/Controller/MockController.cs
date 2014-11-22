using AForge.Neuro;

namespace NeuralTuringMachine.Controller
{
    public class MockController : Network 
    {
        private readonly double[][] _outputs;
        private int _index;

        public MockController(int inputCount, double[][] outputs)
            : base(inputCount + 3, 1)
        {
            _outputs = outputs;
            _index = 0;
        }

        public override double[] Compute(double[] input)
        {
            double[] returnValue = _outputs[_index];
            _index++;
            return returnValue;
        }
    }
}
