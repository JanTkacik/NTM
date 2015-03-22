using System;
using NTM2.Controller;

namespace NTM2.Memory
{
    internal class ReadData
    {
        internal readonly HeadSetting HeadSetting;
        internal readonly Unit[] ReadVector;

        private readonly NTMMemory _controllerMemory;
        private readonly int _cellSize;
        private readonly int _cellCount;

        internal ReadData(HeadSetting headSetting, NTMMemory controllerMemory)
        {
            HeadSetting = headSetting;
            _controllerMemory = controllerMemory;
            _cellSize = _controllerMemory.CellSizeM;
            _cellCount = _controllerMemory.CellCountN;

            ReadVector = new Unit[_cellSize];
            
            for (int i = 0; i < _cellSize; i++)
            {
                double temp = 0;
                for (int j = 0; j < _cellCount; j++)
                {
                    temp += headSetting.AddressingVector[j].Value * controllerMemory.Data[j][i].Value;
                    //if (double.IsNaN(temp))
                    //{
                    //    throw new Exception("Memory error");
                    //}
                }
                ReadVector[i] = new Unit(temp);
            }
        }

        public void BackwardErrorPropagation()
        {
            for (int i = 0; i < _cellCount; i++)
            {
                double gradient = 0;
                Unit[] dataVector = _controllerMemory.Data[i];
                Unit addressingVectorUnit = HeadSetting.AddressingVector[i];
                for (int j = 0; j < _cellSize; j++)
                {
                    double readUnitGradient = ReadVector[j].Gradient;
                    Unit dataUnit = dataVector[j];

                    gradient += readUnitGradient * dataUnit.Value;
                    dataUnit.Gradient += readUnitGradient * addressingVectorUnit.Value;
                }
                addressingVectorUnit.Gradient += gradient;
            }
        }

        #region Factory method

        public static ReadData[] GetVector(int x, Func<int, Tuple<HeadSetting, NTMMemory>> paramGetters)
        {
            ReadData[] vector = new ReadData[x];
            for (int i = 0; i < x; i++)
            {
                Tuple<HeadSetting, NTMMemory> parameters = paramGetters(i);
                vector[i] = new ReadData(parameters.Item1, parameters.Item2);
            }
            return vector;
        }

        #endregion

    }
}
