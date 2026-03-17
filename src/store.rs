use std::{collections::HashMap, hash::Hash};

use super::codebook::Code;

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
