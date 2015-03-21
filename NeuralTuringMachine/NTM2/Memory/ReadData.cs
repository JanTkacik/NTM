using System;
using NTM2.Controller;

namespace NTM2.Memory
{
    internal class ReadData
    {
        private readonly HeadSetting _headSettings;
        private readonly NTMMemory _controllerMemory;
        private readonly Unit[] _data;

        public ReadData(HeadSetting headSettings, NTMMemory controllerMemory)
        {
            _headSettings = headSettings;
            _controllerMemory = controllerMemory;
            _data = new Unit[controllerMemory.Data[0].Length];
            
            for (int i = 0; i < _data.Length; i++)
            {
                double temp = 0;
                for (int j = 0; j < headSettings.Data.Length; j++)
                {
                    temp += headSettings.Data[j].Value * controllerMemory.Data[j][i].Value;
                    if (double.IsNaN(temp))
                    {
                        throw new Exception("Memory error");
                    }
                }
                _data[i] = new Unit(temp);
            }
        }

        public Unit[] Data
        {
            get { return _data; }
        }

        public HeadSetting HeadSetting
        {
            get { return _headSettings; }
        }

        public static ReadData[] GetVector(int x, Func<int,Tuple<HeadSetting, NTMMemory>> paramGetters)
        {
            ReadData[] vector = new ReadData[x];
            for (int i = 0; i < x; i++)
            {
                Tuple<HeadSetting, NTMMemory> parameters = paramGetters(i);
                vector[i] = new ReadData(parameters.Item1, parameters.Item2);
            }
            return vector;
        }

        public void BackwardErrorPropagation()
        {
            for (int i = 0; i < _headSettings.Data.Length; i++)
            {
                double gradient = 0;
                for (int j = 0; j < _data.Length; j++)
                {
                    gradient += _data[j].Gradient*_controllerMemory.Data[i][j].Value;
                }
                _headSettings.Data[i].Gradient += gradient;
            }

            for (int i = 0; i < _controllerMemory.Data.Length; i++)
            {
                for (int j = 0; j < _controllerMemory.Data[i].Length; j++)
                {
                    _controllerMemory.Data[i][j].Gradient += _data[j].Gradient * _headSettings.Data[i].Value;
                }
            }
        }
    }
}
