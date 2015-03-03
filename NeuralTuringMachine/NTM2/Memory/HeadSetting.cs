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

        public HeadSetting(Unit gamma, ShiftedAddressing shiftedAddressing, UnitFactory unitFactory)
        {
            _gamma = gamma;
            _shiftedAddressing = shiftedAddressing;
            _data = unitFactory.GetVector(shiftedAddressing.Data.Length);
            //NO CLUE IN PAPER HOW TO IMPLEMENT - ONLY RESTRICTION IS THAT IT HAS TO BE LARGER THAN 1
            //(Page 9, Part 3.3.2. Focusing by location)
            _gammaIndex = Math.Log(Math.Exp(gamma.Value) + 1) + 1;

            double sum = 0;
            for (int i = 0; i < _data.Length; i++){
                Unit unit = _data[i];
                unit.Value = Math.Pow(_shiftedAddressing.Data[i].Value, _gammaIndex);
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

        public HeadSetting(int memoryColumnsN, ContentAddressing contentAddressing, UnitFactory unitFactory)
        {
            _data = unitFactory.GetVector(memoryColumnsN);
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

        public static HeadSetting[] GetVector(int x, Func<int, Tuple<int, ContentAddressing>> paramGetter, UnitFactory unitFactory)
        {
            HeadSetting[] vector = new HeadSetting[x];
            for (int i = 0; i < x; i++)
            {
                Tuple<int, ContentAddressing> parameters = paramGetter(i);
                vector[i] = new HeadSetting(parameters.Item1, parameters.Item2, unitFactory);
            }
            return vector;
        }

        public void BackwardErrorPropagation()
        {
            //TODO replace by constant
            Unit[] shiftedAddressingData = _shiftedAddressing.Data;
            int shiftedAddressingDataLength = shiftedAddressingData.Length;

            for (int i = 0; i < shiftedAddressingDataLength; i++)
            {
                Unit weight = shiftedAddressingData[i];
                if (weight.Value < double.Epsilon)
                {
                    continue;
                }
                double gradient = 0;
                for (int j = 0; j < _data.Length; j++)
                {
                    Unit dataWeight = _data[j];
                    if (i == j)
                    {
                        gradient += dataWeight.Gradient * (1 - dataWeight.Value);
                    }
                    else
                    {
                        gradient -= dataWeight.Gradient * dataWeight.Value;
                    }
                }
                
                gradient = ((gradient * _gammaIndex) / weight.Value) * _data[i].Value;
                weight.Gradient += gradient;
            }

            double[] lns = new double[shiftedAddressingDataLength];
            double lnexp = 0;
            double s = 0;
            for (int i = 0; i < lns.Length; i++)
            {
                Unit weight = shiftedAddressingData[i];
                if (weight.Value < double.Epsilon)
                {
                    continue;
                }
                lns[i] = Math.Log(weight.Value);
                double pow = Math.Pow(weight.Value, _gammaIndex);
                lnexp += lns[i] * pow;
                s += pow;
            }
            double lnexps = lnexp / s;
            double gradient2 = 0;
            for (int i = 0; i < _data.Length; i++)
            {
                if (shiftedAddressingData[i].Value < double.Epsilon)
                {
                    continue;
                }
                Unit dataWeight = _data[i];
                gradient2 += dataWeight.Gradient * (dataWeight.Value * (lns[i] - lnexps));
            }
            gradient2 = gradient2 / (1 + Math.Exp(-_gamma.Value));
            _gamma.Gradient += gradient2;
        }
    }
}
