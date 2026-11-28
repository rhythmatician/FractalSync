use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    priority: f64,
    cost: f64,
    node: usize,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .total_cmp(&self.priority)
            .then_with(|| other.node.cmp(&self.node))
            .then_with(|| other.cost.total_cmp(&self.cost))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Default)]
pub struct Counts {
    pub expanded: usize,
    pub reopened: usize,
    pub bound_queries: usize,
    pub pruned: usize,
}

impl Counts {
    pub fn record(&mut self, other: Self) {
        self.expanded += other.expanded;
        self.reopened += other.reopened;
        self.bound_queries += other.bound_queries;
        self.pruned += other.pruned;
    }
}

pub struct SearchResult {
    pub cost: Option<f64>,
    pub counts: Counts,
    pub path: Vec<usize>,
}

pub struct SearchLimits {
    pub budget: f64,
    pub prune_with_bound: bool,
}

pub fn search<Edges: IntoIterator<Item = (usize, f64)>>(
    nodes: usize,
    start: usize,
    target: impl Fn(usize) -> bool,
    edges: impl Fn(usize) -> Edges,
    heuristic: impl Fn(usize) -> Option<f64>,
    in_envelope: impl Fn(usize) -> bool,
    limits: SearchLimits,
) -> SearchResult {
    search_with_sum(
        nodes,
        start,
        target,
        edges,
        heuristic,
        in_envelope,
        (limits, |cost, weight| cost + weight),
    )
}

pub(crate) fn search_with_sum<Edges: IntoIterator<Item = (usize, f64)>>(
    nodes: usize,
    start: usize,
    target: impl Fn(usize) -> bool,
    edges: impl Fn(usize) -> Edges,
    heuristic: impl Fn(usize) -> Option<f64>,
    in_envelope: impl Fn(usize) -> bool,
    (limits, accumulate): (SearchLimits, fn(f64, f64) -> f64),
) -> SearchResult {
    let mut queue = BinaryHeap::new();
    let mut costs = vec![f64::INFINITY; nodes];
    let mut previous = vec![usize::MAX; nodes];
    let mut expanded = vec![false; nodes];
    let mut bounds = vec![None; nodes];
    let mut counts = Counts::default();
    let mut push = |queue: &mut BinaryHeap<Candidate>, counts: &mut Counts, node, cost: f64| {
        if cost > limits.budget {
            return;
        }
        if !in_envelope(node) {
            counts.pruned += 1;
            return;
        }
        let lower = *bounds[node].get_or_insert_with(|| heuristic(node).unwrap_or(0.0));
        let priority = (cost + lower).next_down().max(cost);
        if limits.prune_with_bound && priority > limits.budget {
            counts.pruned += 1;
            return;
        }
        queue.push(Candidate {
            priority,
            cost,
            node,
        });
    };
    costs[start] = 0.0;
    push(&mut queue, &mut counts, start, 0.0);
    while let Some(candidate) = queue.pop() {
        let Candidate { node, cost, .. } = candidate;
        if cost > costs[node] {
            continue;
        }
        if target(node) {
            let mut path = vec![node];
            let mut parent = node;
            while parent != start {
                parent = previous[parent];
                path.push(parent);
            }
            path.reverse();
            return SearchResult {
                cost: Some(cost),
                counts,
                path,
            };
        }
        counts.expanded += 1;
        counts.reopened += usize::from(std::mem::replace(&mut expanded[node], true));
        for (neighbour, weight) in edges(node) {
            let new_cost = accumulate(cost, weight);
            if new_cost < costs[neighbour] {
                costs[neighbour] = new_cost;
                previous[neighbour] = node;
                push(&mut queue, &mut counts, neighbour, new_cost);
            }
        }
    }
    SearchResult {
        cost: None,
        counts,
        path: Vec::new(),
    }
}
