using System;
using NTM2.Controller;
using NTM2.Memory.Addressing;

namespace NTM2.Memory
{
    public class HeadSetting
    {
        private readonly Unit[] _data;
        private readonly Unit _gamma;
        //TODO refactor - make sharpening as another class
        private readonly ShiftedAddressing _shiftedAddressing;
        private readonly double _gammaIndex;

        public HeadSetting(Unit gamma, ShiftedAddressing shiftedAddressing)
        {
            _gamma = gamma;
            _shiftedAddressing = shiftedAddressing;
            _data = Unit.GetVector(shiftedAddressing.Data.Length);
            //NO CLUE IN PAPER HOW TO IMPLEMENT - ONLY RESTRICTION IS THAT IT HAS TO BE LARGER THAN 1
            //(Page 9, Part 3.3.2. Focusing by location)
            _gammaIndex = Math.Log(Math.Exp(gamma.Value) + 1) + 1;

            double sum = 0;
            foreach (Unit unit in _data)
            {
                unit.Value = Math.Pow(unit.Value, _gammaIndex);
                sum += unit.Value;
            }

            foreach (Unit unit in _data)
            {
                unit.Value = unit.Value / sum;
                if (double.IsNaN(unit.Value))
                {
                    throw new Exception("Should not be NaN - Error");
                }
            }
        }

        public HeadSetting(int memoryColumnsN, ContentAddressing contentAddressing)
        {
            _data = Unit.GetVector(memoryColumnsN);
            for (int i = 0; i < memoryColumnsN; i++)
            {
                _data[i].Value = contentAddressing.Data[i].Value;
            }
        }

        public Unit[] Data
        {
            get { return _data; }
        }

        public ShiftedAddressing ShiftedAddressing
        {
            get { return _shiftedAddressing; }
        }

        public static HeadSetting[] GetVector(int x, Func<int, Tuple<int, ContentAddressing>> paramGetter)
        {
            HeadSetting[] vector = new HeadSetting[x];
            for (int i = 0; i < x; i++)
            {
                Tuple<int, ContentAddressing> parameters = paramGetter(i);
                vector[i] = new HeadSetting(parameters.Item1, parameters.Item2);
            }
            return vector;
        }

        public void BackwardErrorPropagation()
        {
            for (int i = 0; i < _shiftedAddressing.Data.Length; i++)
            {
                if (_shiftedAddressing.Data[i].Value < double.Epsilon)
                {
                    continue;
                }
                double gradient = 0;
                for (int j = 0; j < _data.Length; j++)
                {
                    if (i == j)
                    {
                        gradient += _data[j].Gradient * (1 - _data[j].Value);
                    }
                    else
                    {
                        gradient -= _data[j].Gradient * _data[j].Value;
                    }
                }
                //PRIORITY ??? 
                gradient = gradient * _gammaIndex / _shiftedAddressing.Data[i].Value * _data[i].Value;
                _shiftedAddressing.Data[i].Gradient += gradient;
            }
            
            double[] lns = new double[_shiftedAddressing.Data.Length];
            double lnexp = 0;
            double s = 0;
            for (int i = 0; i < lns.Length; i++)
            {
                if (_shiftedAddressing.Data[i].Value < double.Epsilon)
                {
                    continue;
                }
                lns[i] = Math.Log(_shiftedAddressing.Data[i].Value);
                double pow = Math.Pow(_shiftedAddressing.Data[i].Value, _gammaIndex);
                lnexp += lns[i] * pow;
                s += pow;
            }
            double lnexps = lnexp / s;
            double gradient2 = 0;
            for (int i = 0; i < _data.Length; i++)
            {
                if (_shiftedAddressing.Data[i].Value < double.Epsilon)
                {
                    continue;
                }
                gradient2 += _data[i].Gradient * (_data[i].Value * (lns[i] - lnexps));
            }
            gradient2 = gradient2 / (1 + Math.Exp(-_gamma.Value));
            _gamma.Gradient += gradient2;
        }
    }
}
