use core::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::codebook::{Code, Scalar, ScoredBooks};
use crate::errors::{TrieError, TrieResult};

#[derive(Debug)]
struct Candidate<'c, T> {
    cumulative_score: T,
    depth: usize,
    node: &'c TrieNode,
    path_idx: Option<usize>,
}

impl<'c, T> Candidate<'c, T> {
    pub fn new(
        cumulative_score: T,
        depth: usize,
        node: &'c TrieNode,
        path_idx: Option<usize>,
    ) -> Self {
        Self {
            cumulative_score,
            depth,
            node,
            path_idx,
        }
    }
}

impl<'c, T: Scalar> PartialEq for Candidate<'c, T> {
    fn eq(&self, other: &Self) -> bool {
        self.cumulative_score == other.cumulative_score
    }
}

impl<'c, T: Scalar> Eq for Candidate<'c, T> {}

impl<'c, T: Scalar> PartialOrd for Candidate<'c, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'c, T: Scalar> Ord for Candidate<'c, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cumulative_score
            .partial_cmp(&other.cumulative_score)
            .unwrap_or(Ordering::Equal)
    }
}

#[derive(Debug, Default)]
struct TrieNode {
    children: HashMap<Code, TrieNode>,
}

// Structural index over code paths for beam prefix search
#[derive(Debug)]
pub struct CodeTrie {
    root: TrieNode,
    pub depth: usize,
}

impl CodeTrie {
    pub fn new(depth: usize) -> Self {
        Self {
            root: TrieNode::default(),
            depth,
        }
    }

    fn traverse<'a>(&'a self, codes: &[Code]) -> TrieResult<&'a TrieNode> {
        if codes.len() != self.depth {
            return Err(TrieError::CodesLengthMismatch(codes.len(), self.depth));
        }

        let mut node = &self.root;
        for code in codes {
            match node.children.get(code) {
                Some(child) => node = child,
                None => return Err(TrieError::NotFound(*code)),
            }
        }
        Ok(node)
    }

    pub fn insert(&mut self, codes: &[Code]) -> TrieResult<()> {
        if codes.len() != self.depth {
            return Err(TrieError::CodesLengthMismatch(codes.len(), self.depth));
        }

        let mut current_node = &mut self.root;
        for &code in codes {
            current_node = current_node.children.entry(code).or_default();
        }
        Ok(())
    }

    pub fn contains(&self, codes: &[Code]) -> bool {
        self.traverse(codes).is_ok()
    }

    fn collect_path(
        path_arena: &[(Code, Option<usize>)],
        mut path_idx: Option<usize>,
        depth: usize,
    ) -> Vec<Code> {
        let mut out = Vec::with_capacity(depth);
        while let Some(i) = path_idx {
            out.push(path_arena[i].0);
            path_idx = path_arena[i].1;
        }
        out.reverse();
        out
    }

    pub fn search<T: Scalar>(
        &self,
        scores: &ScoredBooks<T>,
        k: usize,
    ) -> TrieResult<Vec<Vec<Code>>> {
        if self.depth != scores.num_books {
            return Err(TrieError::BookNumberMismatch(scores.num_books, self.depth));
        }

        let mut result = Vec::with_capacity(k);
        let mut path_arena = Vec::new();
        let mut heap = BinaryHeap::new();

        heap.push(Candidate::new(T::default(), 0, &self.root, None));

        while result.len() < k
            && let Some(candidate) = heap.pop()
        {
            if candidate.depth == self.depth {
                result.push(Self::collect_path(
                    &path_arena,
                    candidate.path_idx,
                    self.depth,
                ));
                continue;
            }

            for (code, &score) in scores.get_book(candidate.depth).iter().enumerate() {
                if let Some(child) = candidate.node.children.get(&code) {
                    path_arena.push((code, candidate.path_idx));
                    let new_path_idx = Some(path_arena.len() - 1);
                    heap.push(Candidate::new(
                        candidate.cumulative_score + score,
                        candidate.depth + 1,
                        child,
                        new_path_idx,
                    ));
                }
            }
        }
        Ok(result)
    }
}
