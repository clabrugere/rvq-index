use std::{collections::HashMap, hash::Hash};

use crate::codebook::Code;

// Bounded traits for Id with blanket implementation
pub trait Id: Default + Copy + Eq + Hash + Send + Sync {}
impl<T: Default + Copy + Eq + Hash + Send + Sync> Id for T {}

#[derive(Debug)]
pub struct EntityStore<I> {
    codes_to_id: HashMap<Vec<Code>, I>,
}

impl<I: Id> EntityStore<I> {
    pub fn new() -> Self {
        Self {
            codes_to_id: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: I, codes: &[Code]) -> Option<I> {
        self.codes_to_id.insert(codes.to_vec(), id)
    }

    pub fn get_id(&self, codes: &[Code]) -> Option<&I> {
        self.codes_to_id.get(codes)
    }

    pub fn len(&self) -> usize {
        self.codes_to_id.len()
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
        assert!(store.insert(id0, &codes0).is_none());
        assert!(store.insert(id1, &codes1).is_none());
        assert!(store.insert(id2, &codes2).is_none());
        assert_eq!(store.len(), 3);
        assert_eq!(store.get_id(&codes0), Some(&id0));
        assert_eq!(store.get_id(&codes1), Some(&id1));
        assert_eq!(store.get_id(&codes2), Some(&id2));
        assert_eq!(store.get_id(&vec![0, 0, 0]), None);
        assert!(store.insert(id1, &codes0).is_some());
    }
}
