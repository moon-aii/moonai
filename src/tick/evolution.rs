use crate::tick::genome::Genome;

pub struct EvolutionManager;

impl EvolutionManager {
    pub const fn seed_initial_population() -> Vec<Genome> {
        Vec::new()
    }

    pub const fn reproduce_population() {}
}
