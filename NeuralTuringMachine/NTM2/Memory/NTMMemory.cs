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
        private readonly double[][] _erasures;

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
            _memoryColumnsN = memory._memoryColumnsN;
            _memoryRowsM = memory._memoryRowsM;
            _headSettings = headSettings;
            _heads = heads;
            _oldMemory = memory;
            _data = Unit.GetTensor2(memory.MemoryColumnsN, memory.MemoryRowsM);

            int headsCount = heads.Length;
            _erase = GetTensor2(headsCount, memory.MemoryRowsM);
            _add = GetTensor2(headsCount, memory.MemoryRowsM);
            _erasures = GetTensor2(memory.MemoryColumnsN, memory.MemoryRowsM);

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

            for (int i = 0; i < _oldMemory.Data.Length; i++)
            {
                Unit[] oldRow = _oldMemory.Data[i];
                double[] erasure = _erasures[i];
                Unit[] row = _data[i];
                for (int j = 0; j < oldRow.Length; j++)
                {
                    Unit oldCell = oldRow[j];
                    double erase = 1;
                    double add = 0;
                    for (int k = 0; k < _headSettings.Length; k++)
                    {
                        HeadSetting headSetting = _headSettings[k];
                        erase *= (1 - (headSetting.Data[i].Value*_erase[k][j]));
                        add += headSetting.Data[i].Value*_add[k][j];
                    }
                    erasure[j] = erase;
                    row[j].Value += (erase*oldCell.Value) + add;
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
                HeadSetting headSetting = _headSettings[i];
                double[] erase = _erase[i];
                double[] add = _add[i];
                for (int j = 0; j < _data.Length; j++)
                {
                    Unit[] row = _data[j];
                    Unit[] oldRow = _oldMemory._data[j];
                    double gradient = 0;
                    for (int k = 0; k < row.Length; k++)
                    {
                        Unit data = row[k];
                        double oldDataValue = oldRow[k].Value;
                        for (int q = 0; q < _headSettings.Length; q++)
                        {
                            if (q == i)
                            {
                                continue; 
                            }
                            HeadSetting setting = _headSettings[q];
                            oldDataValue *= (1 - (setting.Data[j].Value*_erase[q][k]));
                        }
                        gradient += ((oldDataValue*(-erase[k])) + add[k])*data.Gradient;
                    }
                    headSetting.Data[j].Gradient += gradient;
                }
            }

            //Gradient of Erase vector
            for (int k = 0; k < _heads.Length; k++)
            {
                Head head = _heads[k];
                Unit[] headErase = head.EraseVector;
                double[] erase = _erase[k];
                HeadSetting headSetting = _headSettings[k];

                for (int i = 0; i < headErase.Length; i++)
                {
                    double gradient = 0;
                    for (int j = 0; j < _data.Length; j++)
                    {
                        Unit[] row = _data[j];
                        double gradientErase = _oldMemory._data[j][i].Value;
                        for (int q = 0; q < _headSettings.Length; q++)
                        {
                            if (q == k)
                            {
                                continue;
                            }
                            gradientErase *= 1 - (_headSettings[q].Data[j].Value * _erase[q][i]);
                        }
                        gradient += row[i].Gradient * gradientErase * (-headSetting.Data[j].Value);
                    }
                    double e = erase[i];
                    head.EraseVector[i].Gradient += gradient * e * (1 - e);
                }
            }

            //Gradient of Add vector
            for (int k = 0; k < _heads.Length; k++)
            {
                Head head = _heads[k];
                double[] add = _add[k];
                HeadSetting headSetting = _headSettings[k];
                Unit[] addVector = head.AddVector;

                for (int i = 0; i < addVector.Length; i++)
                {
                    double gradient = 0;
                    for (int j = 0; j < _data.Length; j++)
                    {
                        Unit[] row = _data[j];
                        gradient += row[i].Gradient*headSetting.Data[j].Value;
                    }
                    double a = add[i];
                    addVector[i].Gradient += gradient * a * (1 - a);
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
