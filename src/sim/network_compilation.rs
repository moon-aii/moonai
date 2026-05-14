use super::*;

use anyhow::{Context as _, Result, bail};

impl Simulation {
    pub(super) fn compile_population(
        &mut self,
        population_kind: PopulationKind,
        inspected_slot: u32,
    ) -> Result<CompiledNetworkReadbackHeader> {
        check_cuda_status(
            unsafe { dev_compile_population(self.get_dev_state(), population_kind) },
            "moonai_gpu_evolution_compile_population",
        )?;
        check_cuda_status(
            unsafe { dev_write_compiled_header(self.get_dev_state(), population_kind, inspected_slot) },
            "moonai_gpu_evolution_compile_population_header",
        )?;
        device_read(
            "moonai_gpu_evolution_compile_population_header_readback",
            self.device_state.compiled_header_scratch,
        )
    }

    pub(super) fn compile_slot(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<CompiledNetworkReadbackHeader> {
        check_cuda_status(
            unsafe { dev_compile_slot(self.get_dev_state(), population_kind, slot) },
            "moonai_gpu_evolution_compile_slot",
        )?;
        check_cuda_status(
            unsafe { dev_write_compiled_header(self.get_dev_state(), population_kind, slot) },
            "moonai_gpu_evolution_compile_slot_header",
        )?;
        device_read("moonai_gpu_evolution_compile_slot_header_readback", self.device_state.compiled_header_scratch)
    }

    pub(super) fn read_selected_agent_network(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<SelectedAgentNetworkReadback> {
        let status = unsafe { dev_write_selected_agent_network(self.get_dev_state(), population_kind, slot) };
        check_cuda_status(status, "moonai_gpu_evolution_selected_agent_network")?;
        device_read("moonai_gpu_evolution_selected_agent_network_readback", self.device_state.selected_network_scratch)
    }

    pub(super) fn collect_species_summaries(
        &mut self,
        population_kind: PopulationKind,
        max_species: u32,
    ) -> Result<(SpeciesBatchReadbackHeader, Vec<SpeciesSummaryReadback>, Vec<RepresentativeGenomeHeader>)> {
        let species_capacity = usize::try_from(max_species).context("species summary capacity overflowed")?;
        let empty_summary = SpeciesSummaryReadback {
            population_kind,
            species_id: 0,
            size: 0,
            representative_slot: 0,
            avg_complexity: 0.0,
        };
        let empty_representative = RepresentativeGenomeHeader {
            population_kind,
            slot: 0,
            entity_id: 0,
            generation: 0,
            species_id: 0,
            num_nodes: 0,
            num_connections: 0,
        };
        let mut summaries = vec![empty_summary; species_capacity];
        let mut representatives = vec![empty_representative; species_capacity];
        self.classify_species_summaries_impl(population_kind)?;
        let species_count =
            device_read::<u32>("moonai_gpu_evolution_species_summary_count", self.device_state.species_count_scratch)?;
        let returned_species_count = species_count.min(max_species);
        let returned_len =
            usize::try_from(returned_species_count).context("species summary returned length overflowed")?;
        if returned_len > summaries.len() || returned_len > representatives.len() {
            bail!(
                "species summary returned {} entries but the host buffers only allocated {} summary slots and {} representative slots",
                returned_len,
                summaries.len(),
                representatives.len()
            );
        }
        if returned_len > 0 {
            device_read_slice(
                "moonai_gpu_evolution_species_summaries_readback",
                self.device_state.species_summaries_scratch,
                &mut summaries[..returned_len],
            )?;
            device_read_slice(
                "moonai_gpu_evolution_species_representatives_readback",
                self.device_state.representative_headers_scratch,
                &mut representatives[..returned_len],
            )?;
        }
        summaries.truncate(returned_len);
        representatives.truncate(returned_len);
        let header = SpeciesBatchReadbackHeader { population_kind, species_count, returned_species_count };
        Ok((header, summaries, representatives))
    }

    pub(super) fn representative_genome_impl(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<RepresentativeGenomeReadback> {
        let population = match population_kind {
            PopulationKind::Predator => self.device_state.predator,
            PopulationKind::Prey => self.device_state.prey,
        };
        let slot_index = slot as usize;
        let node_stride = population.genome.node_stride as usize;
        let connection_stride = population.genome.connection_stride as usize;
        let entity_id = device_read(
            "moonai_gpu_evolution_representative_genome_entity_id",
            population.entity_id.wrapping_add(slot_index),
        )?;
        let generation = device_read(
            "moonai_gpu_evolution_representative_genome_generation",
            population.generation.wrapping_add(slot_index),
        )?;
        let species_id = device_read(
            "moonai_gpu_evolution_representative_genome_species_id",
            population.species_id.wrapping_add(slot_index),
        )?;
        let num_nodes = device_read(
            "moonai_gpu_evolution_representative_genome_num_nodes",
            population.genome.num_nodes.wrapping_add(slot_index),
        )?;
        let num_connections = device_read(
            "moonai_gpu_evolution_representative_genome_num_connections",
            population.genome.num_connections.wrapping_add(slot_index),
        )?;
        let header = RepresentativeGenomeHeader {
            population_kind,
            slot,
            entity_id,
            generation,
            species_id,
            num_nodes,
            num_connections,
        };
        let returned_nodes = usize::from(header.num_nodes);
        let returned_connections = usize::from(header.num_connections);

        let mut node_types = vec![0_u8; returned_nodes];
        if !node_types.is_empty() {
            let node_base =
                checked_usize_product(slot_index, node_stride, "representative genome node base overflowed")?;
            device_read_slice(
                "moonai_gpu_evolution_representative_genome_node_types",
                population.genome.node_types.wrapping_add(node_base),
                &mut node_types,
            )?;
        }

        let nodes = node_types
            .into_iter()
            .enumerate()
            .map(|(id, node_type)| GenomeNodeReadback { id: id as u32, node_type, reserved0: 0, reserved1: 0 })
            .collect();

        let mut from_nodes = vec![0_i32; returned_connections];
        let mut to_nodes = vec![0_i32; returned_connections];
        let mut weights = vec![0.0_f32; returned_connections];
        let mut innovations = vec![0_u32; returned_connections];
        let mut enabled_flags = vec![0_u8; returned_connections];
        if returned_connections > 0 {
            let connection_base = checked_usize_product(
                slot_index,
                connection_stride,
                "representative genome connection base overflowed",
            )?;
            device_read_slice(
                "moonai_gpu_evolution_representative_genome_connection_from",
                population.genome.connection_from.wrapping_add(connection_base),
                &mut from_nodes,
            )?;
            device_read_slice(
                "moonai_gpu_evolution_representative_genome_connection_to",
                population.genome.connection_to.wrapping_add(connection_base),
                &mut to_nodes,
            )?;
            device_read_slice(
                "moonai_gpu_evolution_representative_genome_connection_weight",
                population.genome.connection_weight.wrapping_add(connection_base),
                &mut weights,
            )?;
            device_read_slice(
                "moonai_gpu_evolution_representative_genome_connection_innovation",
                population.genome.connection_innovation.wrapping_add(connection_base),
                &mut innovations,
            )?;
            device_read_slice(
                "moonai_gpu_evolution_representative_genome_connection_enabled",
                population.genome.connection_enabled.wrapping_add(connection_base),
                &mut enabled_flags,
            )?;
        }

        let connections = from_nodes
            .into_iter()
            .zip(to_nodes)
            .zip(weights)
            .zip(innovations)
            .zip(enabled_flags)
            .map(|((((from_node, to_node), weight), innovation), enabled)| GenomeConnectionReadback {
                from_node,
                to_node,
                weight,
                innovation,
                enabled,
                reserved0: 0,
                reserved1: 0,
            })
            .collect();

        Ok(RepresentativeGenomeReadback { header, nodes, connections })
    }

    pub(super) fn classify_species_summaries_impl(&mut self, population_kind: PopulationKind) -> Result<()> {
        let status = unsafe { dev_classify_species_summaries(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_evolution_species_summaries")
    }
}

unsafe extern "C" {
    fn dev_compile_population(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_compile_slot(state: *mut DeviceState, population_kind: PopulationKind, slot: u32) -> i32;
    fn dev_write_compiled_header(state: *mut DeviceState, population_kind: PopulationKind, slot: u32) -> i32;
    fn dev_write_selected_agent_network(state: *mut DeviceState, population_kind: PopulationKind, slot: u32) -> i32;
    fn dev_classify_species_summaries(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
}
