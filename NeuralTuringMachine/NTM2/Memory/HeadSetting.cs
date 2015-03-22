using System;
using System.Threading.Tasks;
using NTM2.Controller;
using NTM2.Memory.Addressing;
using NTM2.Memory.Addressing.Content;

namespace NTM2.Memory
{
    internal class HeadSetting
    {
        internal readonly Unit[] AddressingVector;
        internal readonly ShiftedAddressing ShiftedVector;

        private readonly Unit _gamma;
        private readonly double _gammaIndex;
        private readonly int _cellCount;
        private readonly Unit[] _shiftedVector;

        internal HeadSetting(Unit gamma, ShiftedAddressing shiftedVector)
        {
            _gamma = gamma;
            ShiftedVector = shiftedVector;
            _shiftedVector = ShiftedVector.ShiftedVector;
            _cellCount = _shiftedVector.Length;
            AddressingVector = UnitFactory.GetVector(_cellCount);

            //NO CLUE IN PAPER HOW TO IMPLEMENT - ONLY RESTRICTION IS THAT IT HAS TO BE LARGER THAN 1
            //(Page 9, Part 3.3.2. Focusing by location)
            _gammaIndex = Math.Log(Math.Exp(gamma.Value) + 1) + 1;

            double sum = 0;
            for (int i = 0; i < _cellCount; i++)
            {
                Unit unit = AddressingVector[i];
                unit.Value = Math.Pow(_shiftedVector[i].Value, _gammaIndex);
                sum += unit.Value;
            }

            foreach (Unit unit in AddressingVector)
            {
                unit.Value = unit.Value / sum;
                if (double.IsNaN(unit.Value))
                {
                    throw new Exception("Should not be NaN - Error");
                }
            }
        }

        internal HeadSetting(int memoryColumnsN, ContentAddressing contentAddressing)
        {
            AddressingVector = UnitFactory.GetVector(memoryColumnsN);
            for (int i = 0; i < memoryColumnsN; i++)
            {
                AddressingVector[i].Value = contentAddressing.ContentVector[i].Value;
            }
        }

        public void BackwardErrorPropagation()
        {
            double[] lns = new double[_cellCount];
            double[] temps = new double[_cellCount];

            double lnexp = 0;
            double s = 0;
            double gradient2 = 0;
            
            Parallel.For(0, _cellCount, ParallelSettings.Options,
                i =>
                {
                    Unit weight = _shiftedVector[i];
                    double weightValue = weight.Value;

                    if (weightValue < double.Epsilon)
                    {
                        return;
                    }
                    double gradient = 0;
                    for (int j = 0; j < _cellCount; j++)
                    {
                        Unit dataWeight = AddressingVector[j];
                        double dataWeightValue = dataWeight.Value;
                        double dataWeightGradient = dataWeight.Gradient;

                        if (i == j)
                        {
                            gradient += dataWeightGradient * (1 - dataWeightValue);
                        }
                        else
                        {
                            gradient -= dataWeightGradient * dataWeightValue;
                        }
                    }

                    gradient = ((gradient * _gammaIndex) / weightValue) * AddressingVector[i].Value;
                    weight.Gradient += gradient;

                    //******************************************************************
                    lns[i] = Math.Log(weightValue);
                    temps[i] = Math.Pow(weightValue, _gammaIndex);
                });
            
            for (int i = 0; i < _cellCount; i++)
            {
                lnexp += lns[i] * temps[i];
                s += temps[i];
            }

            double lnexps = lnexp / s;
            for (int i = 0; i < _cellCount; i++)
            {
                if (_shiftedVector[i].Value < double.Epsilon)
                {
                    continue;
                }
                Unit dataWeight = AddressingVector[i];
                gradient2 += dataWeight.Gradient * (dataWeight.Value * (lns[i] - lnexps));
            }

            gradient2 = gradient2 / (1 + Math.Exp(-_gamma.Value));
            _gamma.Gradient += gradient2;
        }


        #region Factory method

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

        #endregion
    }
}
