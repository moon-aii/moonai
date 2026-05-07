use std::collections::HashMap;

pub struct InnovationTracker {
    next_innovation: u32,
    next_node_id: u32,
    connections: HashMap<(u32, u32), u32>,
}

impl Default for InnovationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl InnovationTracker {
    pub fn new() -> Self {
        Self { next_innovation: 0, next_node_id: 0, connections: HashMap::new() }
    }
}
