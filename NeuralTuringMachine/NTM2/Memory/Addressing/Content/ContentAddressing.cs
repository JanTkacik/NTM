using System;
using System.Linq;
using NTM2.Controller;

namespace NTM2.Memory.Addressing.Content
{
    internal class ContentAddressing
    {
        internal readonly BetaSimilarity[] BetaSimilarities;
        internal readonly Unit[] ContentVector;

        //Implementation of focusing by content (Page 8, Unit 3.3.1 Focusing by Content)
        internal ContentAddressing(BetaSimilarity[] betaSimilarities)
        {
            BetaSimilarities = betaSimilarities;
            ContentVector = UnitFactory.GetVector(betaSimilarities.Length);

            //Subtracting max increase numerical stability
            double max = BetaSimilarities.Max(similarity => similarity.BetaSimilarityMeasure.Value);
            double sum = 0;

            for (int i = 0; i < BetaSimilarities.Length; i++)
            {
                BetaSimilarity unit = BetaSimilarities[i];
                double weight = Math.Exp(unit.BetaSimilarityMeasure.Value - max);
                ContentVector[i].Value = weight;
                sum += weight;
            }
            
            foreach (Unit unit in ContentVector)
            {
                unit.Value = unit.Value/sum;
            }
        }

        internal void BackwardErrorPropagation()
        {
            double gradient = 0;
            foreach (Unit unit in ContentVector)
            {
                gradient += unit.Gradient*unit.Value;
            }

            for (int i = 0; i < ContentVector.Length; i++)
            {
                BetaSimilarities[i].BetaSimilarityMeasure.Gradient += (ContentVector[i].Gradient - gradient)*ContentVector[i].Value;
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
