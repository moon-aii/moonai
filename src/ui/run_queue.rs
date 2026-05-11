use std::collections::VecDeque;
use std::path::PathBuf;

use crate::experiment::SimulationConfig;

#[derive(Debug, Clone, PartialEq)]
pub struct QueuedRun {
    pub id: u64,
    pub experiment_name: String,
    pub simulation_config: SimulationConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RunOutcome {
    Completed,
    Replaced,
    Stopped,
    Failed(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunRecord {
    pub id: u64,
    pub experiment_name: String,
    pub run_name: String,
    pub output_dir: PathBuf,
    pub final_tick: u32,
    pub outcome: RunOutcome,
}

#[derive(Debug, Default, Clone)]
pub struct RunQueue {
    next_id: u64,
    pending: VecDeque<QueuedRun>,
    history: Vec<RunRecord>,
}

impl RunQueue {
    pub const fn reserve_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    pub fn enqueue(&mut self, experiment_name: String, simulation_config: SimulationConfig) -> u64 {
        let id = self.reserve_id();
        self.pending.push_back(QueuedRun { id, experiment_name, simulation_config });
        id
    }

    pub fn pop_next(&mut self) -> Option<QueuedRun> {
        self.pending.pop_front()
    }

    pub const fn pending(&self) -> &VecDeque<QueuedRun> {
        &self.pending
    }

    pub fn history(&self) -> &[RunRecord] {
        &self.history
    }

    pub fn remove_pending(&mut self, id: u64) -> Option<QueuedRun> {
        let index = self.pending.iter().position(|run| run.id == id)?;
        self.pending.remove(index)
    }

    pub fn record(&mut self, record: RunRecord) {
        self.history.push(record);
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

impl RunOutcome {
    pub const fn label(&self) -> &str {
        match self {
            Self::Completed => "completed",
            Self::Replaced => "replaced",
            Self::Stopped => "stopped",
            Self::Failed(_) => "failed",
        }
    }
}
