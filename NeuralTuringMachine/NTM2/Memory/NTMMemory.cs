using System;
using NTM2.Controller;
using NTM2.Memory.Addressing;

namespace NTM2.Memory
{
    public class NTMMemory
    {
        private readonly HeadSetting[] _headSettings;
        private readonly Head[] _heads;
        private readonly NTMMemory _oldMemory;
        private readonly Unit[][] _data;

        private readonly double[][] _erase;
        private readonly double[][] _add;
        private readonly double[][] _tilt;

        private readonly int _memoryColumnsN;
        private readonly int _memoryRowsM;

        public NTMMemory(int memoryColumnsN, int memoryRowsM)
        {
            _memoryColumnsN = memoryColumnsN;
            _memoryRowsM = memoryRowsM;
            _data = Unit.GetTensor2(memoryColumnsN, memoryRowsM);
        }

        public NTMMemory(HeadSetting[] headSettings, Head[] heads, NTMMemory memory)
        {
            _headSettings = headSettings;
            _heads = heads;
            _oldMemory = memory;
            _data = Unit.GetTensor2(memory.MemoryColumnsN, memory.MemoryRowsM);

            int headsCount = heads.Length;
            _erase = GetTensor2(headsCount, memory.MemoryRowsM);
            _add = GetTensor2(headsCount, memory.MemoryRowsM);
            _tilt = GetTensor2(memory.MemoryColumnsN, memory.MemoryRowsM);

            for (int i = 0; i < headsCount; i++)
            {
                Unit[] eraseVector = _heads[i].EraseVector;
                Unit[] addVector = _heads[i].AddVector;

                for (int j = 0; j < _erase[i].Length; j++)
                {
                    _erase[i][j] = Sigmoid.GetValue(eraseVector[j].Value);
                    _add[i][j] = Sigmoid.GetValue(addVector[j].Value);
                }
            }

            for (int i = 0; i < memory.MemoryColumnsN; i++)
            {
                for (int j = 0; j < memory.MemoryRowsM; j++)
                {
                    double mTilt = 1;
                    double adds = 0;
                    for (int k = 0; k < headsCount; k++)
                    {
                        mTilt *= (1 - (_headSettings[k].Data[i].Value*_erase[k][j]));
                        adds += _headSettings[k].Data[i].Value*_add[k][j];
                    }
                    _tilt[i][j] = _oldMemory.Data[i][j].Value * mTilt;
                    Data[i][j].Value = _tilt[i][j] + adds;
                }
            }
        }

        public Unit[][] Data
        {
            get { return _data; }
        }

        public int MemoryColumnsN
        {
            get { return _memoryColumnsN; }
        }

        public int MemoryRowsM
        {
            get { return _memoryRowsM; }
        }

        public HeadSetting[] HeadSettings
        {
            get { return _headSettings; }
        }

        private double[][] GetTensor2(int x, int y)
        {
            double[][] tensor = new double[x][];
            for (int i = 0; i < x; i++)
            {
                tensor[i] = new double[y];
            }
            return tensor;
        }

        public void BackwardErrorPropagation()
        {
            //Gradient of head settings
            for (int i = 0; i < _headSettings.Length; i++)
            {
                for (int j = 0; j < _headSettings[i].Data.Length; j++)
                {
                    double gradient = 0;
                    for (int k = 0; k < _data[j].Length; k++)
                    {
                        double e = _erase[i][k];
                        double gradientErase = _oldMemory._data[j][k].Value * (-e);
                        for (int q = 0; q < _headSettings.Length; q++)
                        {
                            if (q == i)
                            {
                                continue;
                            }
                            gradientErase *= 1 - (_headSettings[q].Data[j].Value*_erase[q][k]);
                        }
                        double gradientAdd = _add[i][k];
                        gradient += (gradientErase + gradientAdd) * _data[j][k].Gradient;
                    }
                    _headSettings[i].Data[j].Gradient += gradient;
                }
            }

            //Gradient of Erase vector
            for (int k = 0; k < _heads.Length; k++)
            {
                Head head = _heads[k];
                for (int i = 0; i < head.EraseVector.Length; i++)
                {
                    double gradient = 0;
                    for (int j = 0; j < _data.Length; j++)
                    {
                        double gradientErase = _oldMemory._data[j][i].Value;
                        for (int q = 0; q < _headSettings.Length; q++)
                        {
                            if (q == k)
                            {
                                continue;
                            }
                            gradientErase *= 1 - (_headSettings[q].Data[j].Value * _erase[q][i]);
                        }
                        gradient += _data[j][i].Gradient * gradientErase * (-_headSettings[k].Data[j].Value);
                    }
                    head.EraseVector[i].Gradient += gradient * _erase[k][i] * (1 - _erase[k][i]);
                }
            }

            //Gradient of Add vector
            for (int k = 0; k < _heads.Length; k++)
            {
                Head head = _heads[k];
                for (int i = 0; i < head.AddVector.Length; i++)
                {
                    double gradient = 0;
                    for (int j = 0; j < _data.Length; j++)
                    {
                        gradient += _data[j][i].Gradient*_headSettings[k].Data[j].Value;
                    }
                    head.AddVector[i].Gradient += gradient*_add[k][i]*(1 - _add[k][i]);
                }
            }

            //Gradient memory
            for (int i = 0; i < _oldMemory._memoryColumnsN; i++)
            {
                for (int j = 0; j < _oldMemory.MemoryRowsM; j++)
                {
                    double gradient = 1;
                    for (int q = 0; q < _headSettings.Length; q++)
                    {
                        gradient *= 1 - (_headSettings[q].Data[i].Value*_erase[q][j]);
                    }
                    _oldMemory._data[i][j].Gradient += gradient*_data[i][j].Gradient;
                }
            }
        }
    }
}
