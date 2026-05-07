#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    Input,
    Hidden,
    Output,
    Bias,
}

#[derive(Debug, Clone)]
pub struct NodeGene {
    pub id: u32,
    pub node_type: NodeType,
}

impl NodeGene {
    pub const fn new(id: u32, node_type: NodeType) -> Self {
        Self { id, node_type }
    }
}

#[derive(Debug, Clone)]
pub struct ConnectionGene {
    pub in_node: u32,
    pub out_node: u32,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: u32,
}

impl ConnectionGene {
    pub const fn new(in_node: u32, out_node: u32, weight: f32, enabled: bool, innovation: u32) -> Self {
        Self { in_node, out_node, weight, enabled, innovation }
    }
}

pub const SENSOR_COUNT: usize = 35;
pub const OUTPUT_COUNT: usize = 2;
pub const INVALID_ENTITY: u32 = u32::MAX;

pub fn deterministic_respawn(seed: u64, tick: u64, entity_id: u32) -> (f32, f32) {
    let mut h = seed;
    h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add((tick << 32) | entity_id as u64);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    let x = (h & 0xFFFF_FFFF) as f32 / u32::MAX as f32;
    let y = ((h >> 32) & 0xFFFF_FFFF) as f32 / u32::MAX as f32;
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_gene_new() {
        let gene = NodeGene::new(0, NodeType::Input);
        assert_eq!(gene.id, 0);
        assert_eq!(gene.node_type, NodeType::Input);
    }

    #[test]
    fn node_type_variants() {
        assert_eq!(NodeType::Input, NodeType::Input);
        assert_eq!(NodeType::Hidden, NodeType::Hidden);
        assert_eq!(NodeType::Output, NodeType::Output);
        assert_eq!(NodeType::Bias, NodeType::Bias);
    }

    #[test]
    fn node_type_equality() {
        let first = NodeType::Hidden;
        let second = NodeType::Hidden;
        let third = NodeType::Output;
        assert_eq!(first, second);
        assert_ne!(first, third);
    }

    #[test]
    fn node_gene_clone() {
        let gene = NodeGene::new(42, NodeType::Hidden);
        let cloned = gene.clone();
        assert_eq!(gene.id, cloned.id);
        assert_eq!(gene.node_type, cloned.node_type);
    }

    #[test]
    fn connection_gene_new() {
        let gene = ConnectionGene::new(0, 1, 0.5, true, 1);
        assert_eq!(gene.in_node, 0);
        assert_eq!(gene.out_node, 1);
        assert_eq!(gene.weight, 0.5);
        assert!(gene.enabled);
        assert_eq!(gene.innovation, 1);
    }

    #[test]
    fn connection_gene_disabled() {
        let gene = ConnectionGene::new(2, 3, -1.0, false, 5);
        assert!(!gene.enabled);
        assert_eq!(gene.weight, -1.0);
    }

    #[test]
    fn connection_gene_clone() {
        let gene = ConnectionGene::new(1, 2, 0.75, true, 3);
        let cloned = gene.clone();
        assert_eq!(gene.in_node, cloned.in_node);
        assert_eq!(gene.out_node, cloned.out_node);
        assert_eq!(gene.weight, cloned.weight);
        assert_eq!(gene.enabled, cloned.enabled);
        assert_eq!(gene.innovation, cloned.innovation);
    }

    #[test]
    fn deterministic_respawn_determinism() {
        let (x1, y1) = deterministic_respawn(42, 100, 5);
        let (x2, y2) = deterministic_respawn(42, 100, 5);
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn deterministic_respawn_different_ticks() {
        let (x1, y1) = deterministic_respawn(42, 100, 5);
        let (x2, y2) = deterministic_respawn(42, 101, 5);
        assert_ne!(x1, x2);
        assert_ne!(y1, y2);
    }

    #[test]
    fn deterministic_respawn_different_entities() {
        let (x1, y1) = deterministic_respawn(42, 100, 5);
        let (x2, y2) = deterministic_respawn(42, 100, 6);
        assert_ne!(x1, x2);
        assert_ne!(y1, y2);
    }

    #[test]
    fn deterministic_respawn_range() {
        for _ in 0..100 {
            let (x, y) = deterministic_respawn(42, 100, 5);
            assert!((0.0..=1.0).contains(&x));
            assert!((0.0..=1.0).contains(&y));
        }
    }
}
