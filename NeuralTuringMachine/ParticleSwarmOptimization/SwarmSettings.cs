namespace ParticleSwarmOptimization
{
    public class SwarmSettings
    {
        public int SwarmSize { get; private set; }
        public double InertiaWeight { get; private set; }
        public double CognitiveWeight { get; private set; }
        public double SocialWeight { get; private set; }
        public double DeathProbability { get; private set; }

        public SwarmSettings(int swarmSize, double inertiaWeight, double cognitiveWeight, double socialWeight, double deathProbability)
        {
            SwarmSize = swarmSize;
            InertiaWeight = inertiaWeight;
            CognitiveWeight = cognitiveWeight;
            SocialWeight = socialWeight;
            DeathProbability = deathProbability;
        }
    }
}
