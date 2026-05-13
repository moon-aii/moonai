use std::fs::{self, File};
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;

use crate::experiment::SimulationConfig;
use crate::sim::{MetricsSummaryReadback, PopulationKind, RepresentativeGenomeReadback, SpeciesSummaryReadback};

pub struct Logger {
    run_dir: PathBuf,
    stats_writer: BufWriter<File>,
    species_writer: BufWriter<File>,
    genomes_path: PathBuf,
    genomes: Vec<GenomeSnapshotRecord>,
}

impl Logger {
    pub fn new(output_dir: &Path, config: &SimulationConfig) -> Result<Self> {
        fs::create_dir_all(output_dir)?;

        let config_path = output_dir.join("config.json");
        let stats_path = output_dir.join("stats.csv");
        let species_path = output_dir.join("species.csv");
        let genomes_path = output_dir.join("genomes.json");

        let config_file = File::create(&config_path)?;
        serde_json::to_writer_pretty(BufWriter::new(config_file), config)?;

        let mut stats_writer = BufWriter::new(File::create(&stats_path)?);
        writeln!(
            stats_writer,
            "tick,predator_count,prey_count,predator_births,prey_births,predator_deaths,prey_deaths,predator_species,prey_species,avg_predator_complexity,avg_prey_complexity,avg_predator_energy,avg_prey_energy,max_predator_generation,avg_predator_generation,max_prey_generation,avg_prey_generation"
        )?;
        stats_writer.flush()?;

        let mut species_writer = BufWriter::new(File::create(&species_path)?);
        writeln!(species_writer, "tick,population,species_id,size,avg_complexity")?;
        species_writer.flush()?;

        let genomes_file = File::create(&genomes_path)?;
        serde_json::to_writer_pretty(BufWriter::new(genomes_file), &Vec::<GenomeSnapshotRecord>::new())?;

        Ok(Self { run_dir: output_dir.to_path_buf(), stats_writer, species_writer, genomes_path, genomes: Vec::new() })
    }

    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    pub fn log_stats(&mut self, summary: &MetricsSummaryReadback) -> Result<()> {
        writeln!(
            self.stats_writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            summary.tick,
            summary.predator_count,
            summary.prey_count,
            summary.predator_births,
            summary.prey_births,
            summary.predator_deaths,
            summary.prey_deaths,
            summary.predator_species,
            summary.prey_species,
            summary.avg_predator_complexity,
            summary.avg_prey_complexity,
            summary.avg_predator_energy,
            summary.avg_prey_energy,
            summary.max_predator_generation,
            summary.avg_predator_generation,
            summary.max_prey_generation,
            summary.avg_prey_generation,
        )?;
        self.stats_writer.flush()?;
        Ok(())
    }

    pub fn log_species(&mut self, tick: u32, summaries: &[SpeciesSummaryReadback]) -> Result<()> {
        for summary in summaries {
            writeln!(
                self.species_writer,
                "{},{},{},{},{}",
                tick,
                population_label(summary.population_kind),
                summary.species_id,
                summary.size,
                summary.avg_complexity,
            )?;
        }
        self.species_writer.flush()?;
        Ok(())
    }

    pub fn log_genomes(&mut self, tick: u32, genomes: &[RepresentativeGenomeReadback]) -> Result<()> {
        self.genomes.extend(genomes.iter().map(|genome| GenomeSnapshotRecord::from_readback(tick, genome)));

        let genomes_file = File::create(&self.genomes_path)?;
        serde_json::to_writer_pretty(BufWriter::new(genomes_file), &self.genomes)?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.stats_writer.flush()?;
        self.species_writer.flush()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct GenomeSnapshotRecord {
    tick: u32,
    population: &'static str,
    slot: u32,
    entity_id: u32,
    generation: u32,
    species_id: u32,
    num_nodes: u16,
    num_connections: u16,
    genome: GenomeTopologyRecord,
}

impl GenomeSnapshotRecord {
    fn from_readback(tick: u32, genome: &RepresentativeGenomeReadback) -> Self {
        Self {
            tick,
            population: population_label(genome.header.population_kind),
            slot: genome.header.slot,
            entity_id: genome.header.entity_id,
            generation: genome.header.generation,
            species_id: genome.header.species_id,
            num_nodes: genome.header.num_nodes,
            num_connections: genome.header.num_connections,
            genome: GenomeTopologyRecord {
                nodes: genome
                    .nodes
                    .iter()
                    .map(|node| GenomeNodeRecord { id: node.id, node_type: node.node_type })
                    .collect(),
                connections: genome
                    .connections
                    .iter()
                    .map(|connection| GenomeConnectionRecord {
                        in_node: connection.from_node,
                        out_node: connection.to_node,
                        weight: connection.weight,
                        innovation: connection.innovation,
                        enabled: connection.enabled != 0,
                    })
                    .collect(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct GenomeTopologyRecord {
    nodes: Vec<GenomeNodeRecord>,
    connections: Vec<GenomeConnectionRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct GenomeNodeRecord {
    id: u32,
    #[serde(rename = "type")]
    node_type: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct GenomeConnectionRecord {
    #[serde(rename = "in")]
    in_node: i32,
    #[serde(rename = "out")]
    out_node: i32,
    weight: f32,
    innovation: u32,
    enabled: bool,
}

const fn population_label(population_kind: PopulationKind) -> &'static str {
    match population_kind {
        PopulationKind::Predator => "predator",
        PopulationKind::Prey => "prey",
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::sim::{
        GenomeConnectionReadback, GenomeNodeReadback, RepresentativeGenomeHeader, RepresentativeGenomeReadback,
    };

    fn temp_run_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_nanos());
        std::env::temp_dir().join(format!("moonai_{name}_{}_{}", std::process::id(), nanos))
    }

    #[test]
    fn logger_writes_expected_artifacts() -> Result<()> {
        let run_dir = temp_run_dir("metrics");
        let config = SimulationConfig::default();
        let mut logger = Logger::new(&run_dir, &config)?;

        logger.log_stats(&MetricsSummaryReadback {
            tick: 1000,
            predator_count: 7,
            prey_count: 11,
            predator_births: 2,
            prey_births: 3,
            predator_deaths: 1,
            prey_deaths: 4,
            predator_species: 2,
            prey_species: 3,
            avg_predator_complexity: 12.5,
            avg_prey_complexity: 8.25,
            avg_predator_energy: 0.61,
            avg_prey_energy: 0.48,
            max_predator_generation: 9,
            avg_predator_generation: 4.0,
            max_prey_generation: 7,
            avg_prey_generation: 3.5,
        })?;
        logger.log_species(
            1000,
            &[
                SpeciesSummaryReadback {
                    population_kind: PopulationKind::Predator,
                    species_id: 4,
                    size: 7,
                    representative_slot: 2,
                    avg_complexity: 13.0,
                },
                SpeciesSummaryReadback {
                    population_kind: PopulationKind::Prey,
                    species_id: 9,
                    size: 11,
                    representative_slot: 5,
                    avg_complexity: 8.5,
                },
            ],
        )?;
        logger.log_genomes(
            1000,
            &[RepresentativeGenomeReadback {
                header: RepresentativeGenomeHeader {
                    population_kind: PopulationKind::Predator,
                    slot: 2,
                    entity_id: 42,
                    generation: 6,
                    species_id: 4,
                    num_nodes: 3,
                    num_connections: 2,
                },
                nodes: vec![
                    GenomeNodeReadback { id: 0, node_type: 0, reserved0: 0, reserved1: 0 },
                    GenomeNodeReadback { id: 1, node_type: 3, reserved0: 0, reserved1: 0 },
                    GenomeNodeReadback { id: 2, node_type: 2, reserved0: 0, reserved1: 0 },
                ],
                connections: vec![
                    GenomeConnectionReadback {
                        from_node: 0,
                        to_node: 2,
                        weight: 0.75,
                        innovation: 8,
                        enabled: 1,
                        reserved0: 0,
                        reserved1: 0,
                    },
                    GenomeConnectionReadback {
                        from_node: 1,
                        to_node: 2,
                        weight: -0.25,
                        innovation: 9,
                        enabled: 0,
                        reserved0: 0,
                        reserved1: 0,
                    },
                ],
            }],
        )?;
        logger.flush()?;

        let stats = fs::read_to_string(run_dir.join("stats.csv"))?;
        let species = fs::read_to_string(run_dir.join("species.csv"))?;
        let config_json = fs::read_to_string(run_dir.join("config.json"))?;
        let genomes_json = fs::read_to_string(run_dir.join("genomes.json"))?;

        assert!(stats.contains("avg_predator_complexity"));
        assert!(stats.contains("1000,7,11,2,3,1,4,2,3,12.5,8.25,0.61,0.48,9,4,7,3.5"));
        assert!(species.contains("1000,predator,4,7,13"));
        assert!(species.contains("1000,prey,9,11,8.5"));
        assert!(config_json.contains("report_interval_ticks"));
        assert!(genomes_json.contains("\"num_nodes\": 3"));
        assert!(genomes_json.contains("\"type\": 0"));
        assert!(genomes_json.contains("\"in\": 0"));
        assert!(genomes_json.contains("\"enabled\": false"));

        fs::remove_dir_all(run_dir)?;
        Ok(())
    }
}
