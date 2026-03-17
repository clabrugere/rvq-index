use crate::codebook::{CodeBooks, Scalar};
use crate::store::Id;

use super::codebook::Code;
use super::errors::RvqIndexResult;
use super::store::EntityStore;
use super::trie::CodeTrie;

#[derive(Debug)]
pub struct RvqIndex<I, T> {
    codebooks: CodeBooks<T>,
    trie: CodeTrie,
    entities: EntityStore<I>,
}

impl<I: Id, T: Scalar> RvqIndex<I, T> {
    pub fn new(codebooks: CodeBooks<T>) -> Self {
        let depth = codebooks.num_books;
        Self {
            codebooks,
            trie: CodeTrie::new(depth),
            entities: EntityStore::new(),
        }
    }

    pub fn insert(&mut self, id: I, codes: &[Code]) -> RvqIndexResult<()> {
        self.trie.insert(codes)?;
        self.entities.insert(id, codes);
        Ok(())
    }

    pub fn search(&self, query: &[T], k: usize) -> RvqIndexResult<Vec<&I>> {
        let scores = self.codebooks.score(query)?;
        let top_k = self
            .trie
            .search(&scores, k)?
            .iter()
            .filter_map(|codes| self.entities.get_id(codes))
            .collect();
        Ok(top_k)
    }

    pub fn get_id(&self, codes: &[Code]) -> Option<&I> {
        self.entities.get_id(codes)
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }
}
