using NTM2.Controller;
using NTM2.Learning;
using NTM2.Memory.Addressing;
using NTM2.Memory.Addressing.Content;

namespace NTM2.Memory
{
    internal class NTMMemory
    {
        internal readonly Unit[][] Data;

        internal readonly HeadSetting[] HeadSettings;
        private readonly Head[] _heads;

        private readonly NTMMemory _oldMemory;
        private readonly BetaSimilarity[][] _oldSimilarities;

        private readonly double[][] _erase;
        private readonly double[][] _add;

        internal readonly int CellCountN;
        internal readonly int CellSizeM;
        internal readonly int HeadCount;

        internal NTMMemory(int cellCountN, int cellSizeM, int headCount)
        {
            CellCountN = cellCountN;
            CellSizeM = cellSizeM;
            HeadCount = headCount;
            Data = UnitFactory.GetTensor2(cellCountN, cellSizeM);
            _oldSimilarities = BetaSimilarity.GetTensor2(headCount, cellCountN);
        }

        internal NTMMemory(HeadSetting[] headSettings, Head[] heads, NTMMemory memory)
        {
            CellCountN = memory.CellCountN;
            CellSizeM = memory.CellSizeM;
            HeadCount = memory.HeadCount;
            HeadSettings = headSettings;
            _heads = heads;
            _oldMemory = memory;
            Data = UnitFactory.GetTensor2(memory.CellCountN, memory.CellSizeM);

            int headsCount = heads.Length;
            _erase = GetTensor2(headsCount, memory.CellSizeM);
            _add = GetTensor2(headsCount, memory.CellSizeM);
            var erasures = GetTensor2(memory.CellCountN, memory.CellSizeM);

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
                double[] erasure = erasures[i];
                Unit[] row = Data[i];
                for (int j = 0; j < oldRow.Length; j++)
                {
                    Unit oldCell = oldRow[j];
                    double erase = 1;
                    double add = 0;
                    for (int k = 0; k < HeadSettings.Length; k++)
                    {
                        HeadSetting headSetting = HeadSettings[k];
                        erase *= (1 - (headSetting.AddressingVector[i].Value*_erase[k][j]));
                        add += headSetting.AddressingVector[i].Value*_add[k][j];
                    }
                    erasure[j] = erase;
                    row[j].Value += (erase*oldCell.Value) + add;
                }
            }
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
            for (int i = 0; i < HeadSettings.Length; i++)
            {
                HeadSetting headSetting = HeadSettings[i];
                double[] erase = _erase[i];
                double[] add = _add[i];
                for (int j = 0; j < Data.Length; j++)
                {
                    Unit[] row = Data[j];
                    Unit[] oldRow = _oldMemory.Data[j];
                    double gradient = 0;
                    for (int k = 0; k < row.Length; k++)
                    {
                        Unit data = row[k];
                        double oldDataValue = oldRow[k].Value;
                        for (int q = 0; q < HeadSettings.Length; q++)
                        {
                            if (q == i)
                            {
                                continue; 
                            }
                            HeadSetting setting = HeadSettings[q];
                            oldDataValue *= (1 - (setting.AddressingVector[j].Value*_erase[q][k]));
                        }
                        gradient += ((oldDataValue*(-erase[k])) + add[k])*data.Gradient;
                    }
                    headSetting.AddressingVector[j].Gradient += gradient;
                }
            }

            //Gradient of Erase vector
            for (int k = 0; k < _heads.Length; k++)
            {
                Head head = _heads[k];
                Unit[] headErase = head.EraseVector;
                double[] erase = _erase[k];
                HeadSetting headSetting = HeadSettings[k];

                for (int i = 0; i < headErase.Length; i++)
                {
                    double gradient = 0;
                    for (int j = 0; j < Data.Length; j++)
                    {
                        Unit[] row = Data[j];
                        double gradientErase = _oldMemory.Data[j][i].Value;
                        for (int q = 0; q < HeadSettings.Length; q++)
                        {
                            if (q == k)
                            {
                                continue;
                            }
                            gradientErase *= 1 - (HeadSettings[q].AddressingVector[j].Value * _erase[q][i]);
                        }
                        gradient += row[i].Gradient * gradientErase * (-headSetting.AddressingVector[j].Value);
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
                HeadSetting headSetting = HeadSettings[k];
                Unit[] addVector = head.AddVector;

                for (int i = 0; i < addVector.Length; i++)
                {
                    double gradient = 0;
                    for (int j = 0; j < Data.Length; j++)
                    {
                        Unit[] row = Data[j];
                        gradient += row[i].Gradient*headSetting.AddressingVector[j].Value;
                    }
                    double a = add[i];
                    addVector[i].Gradient += gradient * a * (1 - a);
                }
            }

            //Gradient memory
            for (int i = 0; i < _oldMemory.CellCountN; i++)
            {
                for (int j = 0; j < _oldMemory.CellSizeM; j++)
                {
                    double gradient = 1;
                    for (int q = 0; q < HeadSettings.Length; q++)
                    {
                        gradient *= 1 - (HeadSettings[q].AddressingVector[i].Value*_erase[q][j]);
                    }
                    _oldMemory.Data[i][j].Gradient += gradient*Data[i][j].Gradient;
                }
            }
        }

        public ContentAddressing[] GetContentAddressing()
        {
            return ContentAddressing.GetVector(HeadCount, i => _oldSimilarities[i]);
        }
        
        public void UpdateWeights(IWeightUpdater weightUpdater)
        {
            foreach (BetaSimilarity[] betaSimilarities in _oldSimilarities)
            {
                foreach (BetaSimilarity betaSimilarity in betaSimilarities)
                {
                    weightUpdater.UpdateWeight(betaSimilarity.BetaSimilarityMeasure);
                }
            }

            weightUpdater.UpdateWeight(Data);
        }

        
    }
}
