use super::*;

use anyhow::{Context as _, Result};

impl Simulation {
    pub(super) fn read_render_snapshot(
        &mut self,
        max_predators: u32,
        max_prey: u32,
        max_food: u32,
    ) -> Result<RenderSnapshotReadback> {
        let mut snapshot = RenderSnapshotReadback::empty();
        self.read_render_snapshot_into(max_predators, max_prey, max_food, &mut snapshot)?;
        Ok(snapshot)
    }

    pub(super) fn read_render_snapshot_into(
        &mut self,
        max_predators: u32,
        max_prey: u32,
        max_food: u32,
        snapshot: &mut RenderSnapshotReadback,
    ) -> Result<()> {
        let empty_predator = RenderAgentReadback {
            population_kind: PopulationKind::Predator,
            slot: 0,
            entity_id: 0,
            species_id: 0,
            generation: 0,
            age: 0.0,
            pos_x: 0.0,
            pos_y: 0.0,
            dir_x: 0.0,
            dir_y: 0.0,
            energy: 0.0,
        };
        let empty_prey = RenderAgentReadback { population_kind: PopulationKind::Prey, ..empty_predator };
        let empty_food = RenderFoodReadback { slot: 0, active: 0, reserved0: 0, reserved1: 0, pos_x: 0.0, pos_y: 0.0 };
        check_cuda_status(
            unsafe { dev_initialize_render_snapshot(self.get_dev_state()) },
            "moonai_gpu_simulation_initialize_render_snapshot",
        )?;
        check_cuda_status(
            unsafe { dev_pack_render_agents(self.get_dev_state(), PopulationKind::Predator, max_predators) },
            "moonai_gpu_simulation_pack_predator_render_snapshot",
        )?;
        check_cuda_status(
            unsafe { dev_pack_render_agents(self.get_dev_state(), PopulationKind::Prey, max_prey) },
            "moonai_gpu_simulation_pack_prey_render_snapshot",
        )?;
        check_cuda_status(
            unsafe { dev_pack_render_food(self.get_dev_state(), max_food) },
            "moonai_gpu_simulation_pack_food_render_snapshot",
        )?;
        let mut header =
            device_read("moonai_gpu_simulation_render_snapshot_header", self.device_state.render_header_scratch)?;
        if header.returned_predators > max_predators {
            header.returned_predators = max_predators;
        }
        if header.returned_prey > max_prey {
            header.returned_prey = max_prey;
        }
        if header.returned_food > max_food {
            header.returned_food = max_food;
        }
        let returned_predators =
            usize::try_from(header.returned_predators).context("predator render length overflowed")?;
        let returned_prey = usize::try_from(header.returned_prey).context("prey render length overflowed")?;
        let returned_food = usize::try_from(header.returned_food).context("food render length overflowed")?;

        snapshot.header = header;
        snapshot.predators.resize(returned_predators, empty_predator);
        snapshot.prey.resize(returned_prey, empty_prey);
        snapshot.food.resize(returned_food, empty_food);

        if returned_predators > 0 {
            device_read_slice(
                "moonai_gpu_simulation_render_snapshot_predators",
                self.device_state.render_predators_scratch,
                &mut snapshot.predators,
            )?;
        }
        if returned_prey > 0 {
            device_read_slice(
                "moonai_gpu_simulation_render_snapshot_prey",
                self.device_state.render_prey_scratch,
                &mut snapshot.prey,
            )?;
        }
        if returned_food > 0 {
            device_read_slice(
                "moonai_gpu_simulation_render_snapshot_food",
                self.device_state.render_food_scratch,
                &mut snapshot.food,
            )?;
        }

        Ok(())
    }
}

unsafe extern "C" {
    fn dev_initialize_render_snapshot(state: *mut DeviceState) -> i32;
    fn dev_pack_render_agents(state: *mut DeviceState, population_kind: PopulationKind, max_count: u32) -> i32;
    fn dev_pack_render_food(state: *mut DeviceState, max_food: u32) -> i32;
}
