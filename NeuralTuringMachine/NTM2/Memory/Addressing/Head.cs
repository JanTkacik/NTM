using System;
using NTM2.Controller;

namespace NTM2.Memory.Addressing
{
    [Serializable]
    public class Head
    {
        private readonly Unit[] _eraseVector;
        private readonly Unit[] _addVector;
        private readonly Unit[] _keyVector;
        private readonly Unit _beta;
        private readonly Unit _gate;
        private readonly Unit _shift;
        private readonly Unit _gama;
        private readonly int _memoryRowSize; //M
        
        public Unit[] KeyVector
        {
            get { return _keyVector; }
        }

        public Unit Beta
        {
            get { return _beta; }
        }

        public Unit Gate
        {
            get { return _gate; }
        }

        public Unit Shift
        {
            get { return _shift; }
        }

        public Unit Gamma
        {
            get { return _gama; }
        }

        public Unit[] EraseVector
        {
            get { return _eraseVector; }
        }

        public Unit[] AddVector
        {
            get { return _addVector; }
        }

        public Head(int memoryRowSize)
        {
            _memoryRowSize = memoryRowSize;
            _eraseVector = UnitFactory.GetVector(memoryRowSize);
            _addVector = UnitFactory.GetVector(memoryRowSize);
            _keyVector = UnitFactory.GetVector(memoryRowSize);
            _beta = new Unit();
            _gate = new Unit();
            _shift = new Unit();
            _gama = new Unit();
        }

        private Head()
        {
            
        }

        public static int GetUnitSize(int memoryRowsM)
        {
            return (3*memoryRowsM) + 4;
        }

        public int GetUnitSize()
        {
            return GetUnitSize(_memoryRowSize);
        }

        public static Head[] GetVector(int length, Func<int, int> constructorParamGetter)
        {
            Head[] vector = new Head[length];
            for (int i = 0; i < length; i++)
            {
                vector[i] = new Head(constructorParamGetter(i));
            }
            return vector;
        }

        public Unit this[int i]
        {
            get
            {
                if (i < _memoryRowSize)
                {
                    return _eraseVector[i];
                }
                if (i < (_memoryRowSize*2))
                {
                    return _addVector[i - _memoryRowSize];
                }
                if (i < (_memoryRowSize*3))
                {
                    return _keyVector[i - (2*_memoryRowSize)];
                }
                if (i == (_memoryRowSize*3))
                {
                    return _beta;
                }
                if (i == (_memoryRowSize * 3) + 1)
                {
                    return _gate;
                }
                if (i == (_memoryRowSize * 3) + 2)
                {
                    return _shift;
                }
                if (i == (_memoryRowSize * 3) + 3)
                {
                    return _gama;
                }
                throw new IndexOutOfRangeException("Index is out of range");
            }
        }
    }
}
