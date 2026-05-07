use std::path::Path;

pub struct Logger;

impl Logger {
    pub const fn new(_output_dir: &Path) -> anyhow::Result<Self> {
        Ok(Self)
    }

    pub const fn log_stats(&mut self, _tick: u64) {}
    pub const fn log_species(&mut self, _tick: u64) {}
    pub const fn log_genomes(&mut self, _tick: u64) {}
}
