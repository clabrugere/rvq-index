use rayon::prelude::*;

use crate::codebook::{Code, CodeBooks, Scalar};
use crate::errors::{RvqIndexError, RvqIndexResult};
use crate::store::{EntityStore, Id};
use crate::trie::CodeTrie;

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
        if codes
            .iter()
            .any(|&code| code >= self.codebooks.num_codes as Code)
        {
            return Err(RvqIndexError::InvalidCode);
        }

        self.trie.insert(codes)?;
        self.entities.insert(id, codes);
        Ok(())
    }

    pub fn insert_many(
        &mut self,
        iterator: impl IntoIterator<Item = (I, Vec<Code>)>,
    ) -> RvqIndexResult<()> {
        for (id, codes) in iterator {
            self.insert(id, &codes)?;
        }
        Ok(())
    }

    // Returns entities corresponding to the top-k scoring code paths. May return more than k entities if collisions exist.
    pub fn search(&self, query: &[T], k: usize) -> RvqIndexResult<Vec<&I>> {
        let scores = self.codebooks.score(query)?;
        let top_k = self
            .trie
            .search(&scores, k)?
            .iter()
            .flat_map(|codes| self.entities.get_ids(codes))
            .collect();
        Ok(top_k)
    }

    pub fn search_batch(&self, queries: &[&[T]], k: usize) -> RvqIndexResult<Vec<Vec<&I>>> {
        queries
            .par_iter()
            .map(|query| self.search(query, k))
            .collect()
    }

    pub fn get_ids(&self, codes: &[Code]) -> &[I] {
        self.entities.get_ids(codes)
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}
