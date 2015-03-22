using System;
using System.Runtime.Serialization;
using NTM2.Controller;

namespace NTM2.Memory.Addressing.Content
{
    //This class implements equation from page 8 - _b i exped to ensure that it will be positive
    [DataContract]
    internal class BetaSimilarity
    {
        [DataMember]
        private readonly Unit _beta;
        [DataMember]
        internal readonly SimilarityMeasure Similarity;
        [DataMember]
        internal readonly Unit BetaSimilarityMeasure;
        //Key strength beta
        [DataMember]
        private readonly double _b;

        internal BetaSimilarity(Unit beta, SimilarityMeasure similarity)
        {
            _beta = beta;
            Similarity = similarity;
            //Ensuring that beta will be positive
            _b = Math.Exp(_beta.Value);
            BetaSimilarityMeasure = new Unit(_b * Similarity.Similarity.Value);
        }

        internal BetaSimilarity()
        {
            _beta = new Unit();
            BetaSimilarityMeasure = new Unit();
        }
        
        internal void BackwardErrorPropagation()
        {
            Unit similarity = Similarity.Similarity;
            double betaGradient = BetaSimilarityMeasure.Gradient;

            _beta.Gradient += similarity.Value * _b * betaGradient;
            similarity.Gradient += _b * betaGradient;
        }

        #region Factory methods

        internal static BetaSimilarity[][] GetTensor2(int x, int y)
        {
            BetaSimilarity[][] tensor = new BetaSimilarity[x][];
            for (int i = 0; i < x; i++)
            {
                tensor[i] = GetVector(y);
            }
            return tensor;
        }

        internal static BetaSimilarity[] GetVector(int x)
        {
            BetaSimilarity[] vector = new BetaSimilarity[x];
            for (int i = 0; i < x; i++)
            {
                vector[i] = new BetaSimilarity();
            }
            return vector;
        } 

        #endregion
    }
}
