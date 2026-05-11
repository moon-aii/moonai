use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

thread_local! {
    static ACTIVE_PROFILER: RefCell<Option<Rc<RefCell<ProfilerState>>>> = const { RefCell::new(None) };
    static ACTIVE_SCOPE_STACK: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

#[derive(Debug, Clone, Default)]
pub struct Profiler {
    state: Rc<RefCell<ProfilerState>>,
}

impl Profiler {
    pub fn bind(&self) -> ProfilerSession {
        let previous_profiler = ACTIVE_PROFILER.with(|slot| slot.replace(Some(Rc::clone(&self.state))));
        let previous_stack = ACTIVE_SCOPE_STACK.with(|stack| std::mem::take(&mut *stack.borrow_mut()));
        ProfilerSession { previous_profiler, previous_stack }
    }

    pub fn formatted_rows(&self, root_name: &str, current_tick: u32) -> Vec<String> {
        self.state.borrow().formatted_rows(root_name, current_tick)
    }
}

pub struct ProfilerSession {
    previous_profiler: Option<Rc<RefCell<ProfilerState>>>,
    previous_stack: Vec<&'static str>,
}

impl Drop for ProfilerSession {
    fn drop(&mut self) {
        ACTIVE_PROFILER.with(|slot| {
            let _ = slot.replace(self.previous_profiler.take());
        });
        ACTIVE_SCOPE_STACK.with(|stack| {
            *stack.borrow_mut() = std::mem::take(&mut self.previous_stack);
        });
    }
}

pub struct ScopeGuard {
    profiler: Option<Rc<RefCell<ProfilerState>>>,
    path: Vec<&'static str>,
    name: &'static str,
    started_at: Instant,
}

impl ScopeGuard {
    pub fn enter(name: &'static str) -> Self {
        let profiler = ACTIVE_PROFILER.with(|slot| slot.borrow().clone());
        let path = if profiler.is_some() {
            ACTIVE_SCOPE_STACK.with(|stack| {
                let mut stack = stack.borrow_mut();
                stack.push(name);
                stack.clone()
            })
        } else {
            Vec::new()
        };

        Self { profiler, path, name, started_at: Instant::now() }
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        if let Some(profiler) = &self.profiler {
            profiler.borrow_mut().record_scope(&self.path, self.started_at.elapsed());
            ACTIVE_SCOPE_STACK.with(|stack| {
                let popped = stack.borrow_mut().pop();
                debug_assert_eq!(popped, Some(self.name));
            });
        }
    }
}

#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _profile_scope_guard = $crate::profiler::ScopeGuard::enter($name);
    };
}

#[derive(Debug, Default)]
struct ProfilerState {
    roots: HashMap<String, ProfileNode>,
    root_order: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct ProfileNode {
    total: Duration,
    children: HashMap<String, ProfileNode>,
    child_order: Vec<String>,
}

#[derive(Debug)]
struct ProfileRow {
    label: String,
    percent: f64,
    per_tick_us: u64,
}

impl ProfilerState {
    fn record_scope(&mut self, path: &[&'static str], elapsed: Duration) {
        let Some((name, rest)) = path.split_first() else {
            return;
        };

        let node = Self::ensure_node(&mut self.roots, &mut self.root_order, name);
        if rest.is_empty() {
            node.total += elapsed;
            return;
        }

        Self::record_child_scope(node, rest, elapsed);
    }

    fn record_child_scope(node: &mut ProfileNode, path: &[&'static str], elapsed: Duration) {
        let child = Self::ensure_node(&mut node.children, &mut node.child_order, path[0]);
        if path.len() == 1 {
            child.total += elapsed;
            return;
        }

        Self::record_child_scope(child, &path[1..], elapsed);
    }

    fn ensure_node<'a>(
        nodes: &'a mut HashMap<String, ProfileNode>,
        order: &mut Vec<String>,
        name: &str,
    ) -> &'a mut ProfileNode {
        let key = name.to_owned();
        if !nodes.contains_key(name) {
            order.push(key.clone());
        }
        nodes.entry(key).or_default()
    }

    fn formatted_rows(&self, root_name: &str, current_tick: u32) -> Vec<String> {
        let Some(root) = self.roots.get(root_name) else {
            return Vec::new();
        };

        let mut rows = Vec::new();
        Self::collect_rows(root_name, root, 0, root.total, current_tick, &mut rows);
        let label_width = rows.iter().map(|row| row.label.len()).max().unwrap_or(root_name.len()).max(16);
        let micros_width = rows.iter().map(|row| row.per_tick_us.to_string().len()).max().unwrap_or(1).max(2);

        rows.into_iter()
            .map(|row| {
                let percent = format!("{:.0}%", row.percent);
                format!(
                    "{:<label_width$} {:>4} {:>micros_width$} us",
                    row.label,
                    percent,
                    row.per_tick_us,
                    label_width = label_width,
                    micros_width = micros_width,
                )
            })
            .collect()
    }

    fn collect_rows(
        name: &str,
        node: &ProfileNode,
        depth: usize,
        root_total: Duration,
        current_tick: u32,
        rows: &mut Vec<ProfileRow>,
    ) {
        let percent = if depth == 0 {
            if root_total > Duration::ZERO { 100.0 } else { 0.0 }
        } else if root_total > Duration::ZERO {
            (node.total.as_secs_f64() / root_total.as_secs_f64()) * 100.0
        } else {
            0.0
        };
        let per_tick_us = if current_tick == 0 {
            0
        } else {
            ((node.total.as_secs_f64() * 1_000_000.0) / f64::from(current_tick)).round() as u64
        };

        rows.push(ProfileRow { label: format!("{}{}", "  ".repeat(depth), name), percent, per_tick_us });

        for child_name in &node.child_order {
            if let Some(child) = node.children.get(child_name) {
                Self::collect_rows(child_name, child, depth + 1, root_total, current_tick, rows);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_nested_scopes(profiler: &Profiler) {
        let _session = profiler.bind();
        {
            crate::profile_scope!("frame");
            {
                crate::profile_scope!("tick");
                {
                    crate::profile_scope!("spatial_grid");
                }
                {
                    crate::profile_scope!("reproduction");
                }
            }
        }
    }

    #[test]
    fn profiler_records_nested_scope_tree() {
        let profiler = Profiler::default();

        with_nested_scopes(&profiler);

        let state = profiler.state.borrow();
        let frame = state.roots.get("frame").expect("frame scope missing");
        let tick = frame.children.get("tick").expect("tick scope missing");

        assert!(frame.total >= Duration::ZERO);
        assert!(tick.children.contains_key("spatial_grid"));
        assert!(tick.children.contains_key("reproduction"));
        assert_eq!(frame.child_order, vec!["tick"]);
        assert_eq!(tick.child_order, vec!["spatial_grid", "reproduction"]);
    }

    #[test]
    fn formatted_rows_use_indentation_and_alignment() {
        let mut state = ProfilerState::default();
        state.record_scope(&["frame"], Duration::from_micros(1000));
        state.record_scope(&["frame", "tick"], Duration::from_micros(500));
        state.record_scope(&["frame", "tick", "spatial_grid"], Duration::from_micros(80));
        state.record_scope(&["frame", "tick", "reproduction"], Duration::from_micros(150));

        let profiler = Profiler { state: Rc::new(RefCell::new(state)) };
        let rows = profiler.formatted_rows("frame", 10);

        assert_eq!(rows.len(), 4);
        assert!(rows[0].starts_with("frame"));
        assert!(rows[1].starts_with("  tick"));
        assert!(rows[2].starts_with("    spatial_grid"));
        assert!(rows[3].starts_with("    reproduction"));
        assert!(rows[0].contains("100%"));
        assert!(rows[1].contains("50%"));
    }
}
