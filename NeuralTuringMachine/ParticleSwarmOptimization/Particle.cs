using System;

namespace ParticleSwarmOptimization
{
    public class Particle
    {
        private const double VelocityRatio = 0.1;
        public double[] Position { get; private set; }
        public double Error { get; private set; }
        public double[] Velocity { get; private set; }
        public double[] BestPosition { get; private set; }
        public double BestError { get; private set; }
        private readonly Random _randomGenerator;
        private readonly IErrorFunction _errorFunction;
        
        public Particle(IErrorFunction errorFunction)
        {
            _errorFunction = errorFunction;
            _randomGenerator = new Random(DateTime.Now.Millisecond);

            Position = new double[errorFunction.Dimensions];
            Velocity = new double[errorFunction.Dimensions];
            BestPosition = new double[errorFunction.Dimensions];

            Birth();
        }

        private void Birth()
        {
            for (int i = 0; i < _errorFunction.Dimensions; i++)
            {
                double min = _errorFunction.MinForDimension(i);
                double max = _errorFunction.MaxForDimension(i);
                Position[i] = NextRandomInInterval(min, max);
                Velocity[i] = NextRandomInInterval(min*VelocityRatio, max*VelocityRatio);
            }

            Error = _errorFunction.CalculateError(Position);
            BestError = Error;
            Position.CopyTo(BestPosition, 0);
        }

        private double NextRandomInInterval(double min, double max)
        {
            return ((max - min)*_randomGenerator.NextDouble()) + min;
        }

        public void Update(double[] bestGlobalPosition, SwarmSettings swarmSettings)
        {
            double die = _randomGenerator.NextDouble();

            if (die >= swarmSettings.DeathProbability)
            {
                int dimensions = _errorFunction.Dimensions;
                double inertiaWeight = swarmSettings.InertiaWeight;
                double cognitiveWeight = swarmSettings.CognitiveWeight;
                double socialWeight = swarmSettings.SocialWeight;

                for (int i = 0; i < dimensions; i++)
                {
                    double randomCongintive = _randomGenerator.NextDouble();
                    double randomSocial = _randomGenerator.NextDouble();

                    Velocity[i] = (inertiaWeight*Velocity[i]) +
                                  (cognitiveWeight*randomCongintive*(BestPosition[i] - Position[i])) +
                                  (socialWeight*randomSocial*(bestGlobalPosition[i] - Position[i]));
                    Position[i] = Position[i] + Velocity[i];
                    if (Position[i] < _errorFunction.MinForDimension(i))
                    {
                        Position[i] = _errorFunction.MinForDimension(i);
                    }
                    if (Position[i] > _errorFunction.MaxForDimension(i))
                    {
                        Position[i] = _errorFunction.MaxForDimension(i);
                    }
                }

                Error = _errorFunction.CalculateError(Position);
                if (Error < BestError)
                {
                    BestError = Error;
                }
                Position.CopyTo(BestPosition, 0);
            }
            else
            {
                Birth();
            }
        }
    }
}
