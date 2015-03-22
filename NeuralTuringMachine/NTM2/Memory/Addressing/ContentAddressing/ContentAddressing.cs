using System;
using System.Linq;
using NTM2.Controller;

namespace NTM2.Memory.Addressing.ContentAddressing
{
    internal class ContentAddressing
    {
        internal readonly BetaSimilarity[] BetaSimilarities;
        internal readonly Unit[] Data;

        //Implementation of focusing by content (Page 8, Unit 3.3.1 Focusing by Content)
        internal ContentAddressing(BetaSimilarity[] betaSimilarities)
        {
            BetaSimilarities = betaSimilarities;
            Data = UnitFactory.GetVector(betaSimilarities.Length);

            //Subtracting max increase numerical stability
            double max = BetaSimilarities.Max(similarity => similarity.BetaSimilarityMeasure.Value);
            double sum = 0;

            for (int i = 0; i < BetaSimilarities.Length; i++)
            {
                BetaSimilarity unit = BetaSimilarities[i];
                double weight = Math.Exp(unit.BetaSimilarityMeasure.Value - max);
                Data[i].Value = weight;
                sum += weight;
            }
            
            foreach (Unit unit in Data)
            {
                unit.Value = unit.Value/sum;
            }
        }

        internal void BackwardErrorPropagation()
        {
            double gradient = 0;
            foreach (Unit unit in Data)
            {
                gradient += unit.Gradient*unit.Value;
            }

            for (int i = 0; i < Data.Length; i++)
            {
                BetaSimilarities[i].BetaSimilarityMeasure.Gradient += (Data[i].Gradient - gradient)*Data[i].Value;
            }
        }

        #region Factory method

        internal static ContentAddressing[] GetVector(int x, Func<int, BetaSimilarity[]> paramGetter)
        {
            ContentAddressing[] vector = new ContentAddressing[x];
            for (int i = 0; i < x; i++)
            {
                vector[i] = new ContentAddressing(paramGetter(i));
            }
            return vector;
        } 

        #endregion
    }
}
