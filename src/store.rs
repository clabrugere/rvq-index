use std::{collections::HashMap, hash::Hash};

use crate::codebook::Code;

// Bounded traits for Id with blanket implementation
pub trait Id: Default + Copy + Eq + Hash + Send + Sync {}
impl<T: Default + Copy + Eq + Hash + Send + Sync> Id for T {}

#[derive(Debug, Default)]
pub struct EntityStore<I> {
    codes_to_id: HashMap<Vec<Code>, Vec<I>>,
}

impl<I: Id> EntityStore<I> {
    pub fn new() -> Self {
        Self {
            codes_to_id: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: I, codes: &[Code]) {
        self.codes_to_id.entry(codes.to_vec()).or_default().push(id);
    }

    pub fn get_ids(&self, codes: &[Code]) -> &[I] {
        self.codes_to_id
            .get(codes)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.codes_to_id.values().map(Vec::len).sum()
    }

    pub fn count_colliding_ids(&self) -> usize {
        self.codes_to_id
            .values()
            .filter(|ids| ids.len() > 1)
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.codes_to_id.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_store() {
        let mut store = EntityStore::new();

        let (codes0, id0) = (vec![1, 2, 3], 0);
        let (codes1, id1) = (vec![4, 5, 6], 1);
        let (codes2, id2) = (vec![7, 8, 9], 2);

        assert_eq!(store.len(), 0);

        store.insert(id0, &codes0);
        store.insert(id1, &codes1);
        store.insert(id2, &codes2);
        assert_eq!(store.len(), 3);

        assert_eq!(store.get_ids(&codes0), &[id0]);
        assert_eq!(store.get_ids(&codes1), &[id1]);
        assert_eq!(store.get_ids(&codes2), &[id2]);
    }

    #[test]
    fn test_entity_store_collisions() {
        let mut store = EntityStore::new();

        let (codes0, id0) = (vec![1, 2, 3], 0);
        let (codes1, id1) = (vec![1, 2, 3], 1);
        let (codes2, id2) = (vec![4, 5, 6], 2);

        store.insert(id0, &codes0);
        store.insert(id1, &codes1);
        store.insert(id2, &codes2);

        assert_eq!(store.len(), 3);
        assert_eq!(store.count_colliding_ids(), 1);
    }

    #[test]
    fn test_entity_store_unknown_codes() {
        let mut store = EntityStore::new();

        let (codes0, id0) = (vec![1, 2, 3], 0);
        let (codes1, id1) = (vec![4, 5, 6], 1);

        store.insert(id0, &codes0);
        store.insert(id1, &codes1);

        assert_eq!(store.get_ids(&[7, 8, 9]).len(), 0);
    }
}
