using System;
using System.Collections.Generic;
using CsvHelper;

namespace NeuralTuringMachine.Controller
{
    public class ControllerInput
    {
        private readonly int _dataInputLength;
        public double[] Input { get; private set; }
        
        public ControllerInput(double[] dataInput, double[] readHeadInput, int controllerInputLength)
        {
            _dataInputLength = dataInput.Length;
            Input = new double[controllerInputLength];
            Array.Copy(dataInput, Input, _dataInputLength);
            int offset = _dataInputLength;
            Array.Copy(readHeadInput, 0, Input, offset, readHeadInput.Length);
        }

        public ControllerInput(double[] dataInput, IEnumerable<double[]> readHeadInput, int controllerInputLength)
        {
            int dataInputLength = dataInput.Length;
            Input = new double[controllerInputLength];
            Array.Copy(dataInput, Input, dataInputLength);
            int offset = dataInputLength;
            foreach (double[] array in readHeadInput)
            {
                Array.Copy(array, 0, Input, offset, array.Length);
                offset += array.Length;
            }
        }

        public ControllerInput(double[] dataInput, int controllerInputLength)
        {
            Random rand = new Random(DateTime.Now.Millisecond);
            Input = new double[controllerInputLength];
            Array.Copy(dataInput, Input, dataInput.Length);
            int offset = dataInput.Length;
            for (int i = offset; i < controllerInputLength; i++)
            {
                Input[i] = rand.NextDouble();
            }
        }

        public void WriteCSVLog(CsvWriter logger)
        {
            logger.WriteField("Input");
            logger.WriteField("Input-Data");
            for (int i = 0; i < _dataInputLength; i++)
            {
                logger.WriteField(Input[i]);
            }
            logger.WriteField("Input-Read");
            for (int i = _dataInputLength; i < Input.Length; i++)
            {
                logger.WriteField(Input[i]);
            }
        }
    }
}
