using System;
using NTM2.Controller;

namespace NTM2.Memory.Addressing
{
    internal class ShiftedAddressing
    {
        private readonly Unit _shift;
        private readonly Unit[] _gatedVector;
        private readonly int _convolution;
        private readonly int _cellCount;
        private readonly double _simj;
        private readonly double _oneMinusSimj;
        private readonly double _shiftWeight;
        
        internal readonly GatedAddressing GatedAddressing;
        internal readonly Unit[] ShiftedVector;

        //IMPLEMENTATION OF SHIFT - page 9
        internal ShiftedAddressing(Unit shift, GatedAddressing gatedAddressing)
        {
            _shift = shift;
            GatedAddressing = gatedAddressing;
            _gatedVector = GatedAddressing.GatedVector;
            _cellCount = _gatedVector.Length;
            
            ShiftedVector = UnitFactory.GetVector(_cellCount);
            double cellCountDbl = _cellCount;

            //Max shift is from range -1 to 1
            _shiftWeight = Sigmoid.GetValue(_shift.Value);
            double maxShift = ((2 * _shiftWeight) - 1);
            double convolutionDbl = (maxShift + cellCountDbl) % cellCountDbl;

            _simj = 1 - (convolutionDbl - Math.Floor(convolutionDbl));
            _oneMinusSimj = (1 - _simj);
            _convolution = (int)convolutionDbl;
            
            for (int i = 0; i < _cellCount; i++)
            {
                int imj = (i + _convolution) % _cellCount;

                Unit vectorItem = ShiftedVector[i];

                vectorItem.Value = (_gatedVector[imj].Value * _simj) +
                                   (_gatedVector[(imj + 1) % _cellCount].Value * _oneMinusSimj);
                if (vectorItem.Value < 0 || double.IsNaN(vectorItem.Value))
                {
                    throw new Exception("Error - weight should not be smaller than zero or nan");
                }
            }
        }

        internal void BackwardErrorPropagation()
        {
            double gradient = 0;
            for (int i = 0; i < _cellCount; i++)
            {
                Unit vectorItem = ShiftedVector[i];
                int imj = (i + (_convolution)) % _cellCount;
                gradient += ((-_gatedVector[imj].Value) + _gatedVector[(imj + 1) % _cellCount].Value) * vectorItem.Gradient;
                int j = (i - (_convolution) + _cellCount) % _cellCount;
                _gatedVector[i].Gradient += (vectorItem.Gradient * _simj) + (ShiftedVector[(j - 1 + _cellCount) % _cellCount].Gradient * _oneMinusSimj);
            }

            gradient = gradient * 2 * _shiftWeight * (1 - _shiftWeight);
            _shift.Gradient += gradient;
        }
    }
}
