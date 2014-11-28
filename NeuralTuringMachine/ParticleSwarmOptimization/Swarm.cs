namespace ParticleSwarmOptimization
{
    public class Swarm
    {
        private readonly SwarmSettings _swarmSettings;
        private double _bestGlobalError;
        private readonly double[] _bestGlobalPosition;
        private readonly Particle[] _swarm;

        public Swarm(IErrorFunction errorFunction, SwarmSettings swarmSettings)
        {
            _swarmSettings = swarmSettings;
            _swarm = new Particle[_swarmSettings.SwarmSize];
            _bestGlobalPosition = new double[errorFunction.Dimensions];
            _bestGlobalError = double.PositiveInfinity;
            for (int i = 0; i < _swarmSettings.SwarmSize; i++)
            {
                _swarm[i] = new Particle(errorFunction);
                if (_swarm[i].BestError < _bestGlobalError)
                {
                    _bestGlobalError = _swarm[i].BestError;
                    _swarm[i].Position.CopyTo(_bestGlobalPosition, 0);
                }
            }
        }

        public void RunEpoch()
        {
            foreach (Particle particle in _swarm)
            {
                particle.Update(_bestGlobalPosition, _swarmSettings);
                if (particle.BestError < _bestGlobalError)
                {
                    _bestGlobalError = particle.BestError;
                    particle.BestPosition.CopyTo(_bestGlobalPosition, 0);
                }
            }
        }
    }
}
