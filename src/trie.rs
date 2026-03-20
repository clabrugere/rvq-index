use std::{cmp::Ordering, collections::BinaryHeap};

use crate::codebook::{Code, Scalar, ScoredBooks};
use crate::errors::{TrieError, TrieResult};

#[derive(Debug)]
struct Candidate<'c, T> {
    upper_bound: T,
    cumulative_score: T,
    depth: usize,
    node: &'c TrieNode,
    path_idx: Option<usize>,
}

impl<'c, T> Candidate<'c, T> {
    pub fn new(
        upper_bound: T,
        cumulative_score: T,
        depth: usize,
        node: &'c TrieNode,
        path_idx: Option<usize>,
    ) -> Self {
        Self {
            upper_bound,
            cumulative_score,
            depth,
            node,
            path_idx,
        }
    }
}

impl<T: Scalar> PartialEq for Candidate<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.upper_bound == other.upper_bound
            && self.cumulative_score == other.cumulative_score
            && self.depth == other.depth
            && self.path_idx == other.path_idx
    }
}

impl<T: Scalar> Eq for Candidate<'_, T> {}

impl<T: Scalar> PartialOrd for Candidate<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Scalar> Ord for Candidate<'_, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.upper_bound
            .partial_cmp(&other.upper_bound)
            .unwrap_or(Ordering::Equal)
    }
}

#[derive(Debug, Default)]
struct TrieNode {
    children: Vec<Option<Self>>,
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

    pub fn is_empty(&self) -> bool {
        self.root.children.is_empty()
    }

    fn traverse<'n>(&'n self, codes: &[Code]) -> TrieResult<&'n TrieNode> {
        let mut node = &self.root;
        for code in codes {
            match node.children.get(*code).and_then(Option::as_ref) {
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
            if current_node.children.len() <= code {
                current_node.children.resize_with(code + 1, || None);
            }
            current_node = current_node.children[code].get_or_insert_with(TrieNode::default);
        }
        Ok(())
    }

    pub fn contains(&self, codes: &[Code]) -> bool {
        if codes.len() != self.depth {
            return false;
        }
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

        // max achievable score from book to the end
        let mut remaining_max = vec![T::default(); self.depth + 1];
        for book in (0..self.depth).rev() {
            remaining_max[book] = remaining_max[book + 1] + scores.get_book_max(book);
        }

        let mut result = Vec::with_capacity(k);
        let mut path_arena = Vec::new();
        let mut heap = BinaryHeap::new();

        heap.push(Candidate::new(
            remaining_max[0],
            T::default(),
            0,
            &self.root,
            None,
        ));

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
                if let Some(child) = candidate.node.children.get(code).and_then(Option::as_ref) {
                    path_arena.push((code, candidate.path_idx));

                    let new_path_idx = Some(path_arena.len() - 1);
                    let cumulative_score = candidate.cumulative_score + score;
                    let next_depth = candidate.depth + 1;
                    let upper_bound = cumulative_score + remaining_max[next_depth];

                    heap.push(Candidate::new(
                        upper_bound,
                        cumulative_score,
                        next_depth,
                        child,
                        new_path_idx,
                    ));
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::ScoredBooks;

    // book 0 scores: [1.0, 2.0, 3.0]
    // book 1 scores: [10.0, 20.0, 30.0]
    // paths:  [0,0] -> 11,  [0,2] -> 31,  [1,1] -> 22
    fn make_trie_and_scores() -> (CodeTrie, ScoredBooks<f32>) {
        let mut trie = CodeTrie::new(2);
        trie.insert(&[0, 0]).unwrap();
        trie.insert(&[0, 2]).unwrap();
        trie.insert(&[1, 1]).unwrap();
        let scores = ScoredBooks::new(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 2, 3);
        (trie, scores)
    }

    #[test]
    fn test_insertion() {
        let mut trie = CodeTrie::new(3);
        assert!(trie.is_empty());

        let codes1 = vec![1, 2, 3];
        let codes2 = vec![1, 2, 4];
        let codes3 = vec![1, 3, 4];

        trie.insert(&codes1).unwrap();
        trie.insert(&codes2).unwrap();
        trie.insert(&codes3).unwrap();

        assert!(trie.contains(&codes1));
        assert!(trie.contains(&codes2));
        assert!(trie.contains(&codes3));
        assert!(!trie.contains(&vec![0, 0, 0]));
    }

    #[test]
    fn test_insert_duplicate() {
        let mut trie = CodeTrie::new(2);
        trie.insert(&[1, 2]).unwrap();
        trie.insert(&[1, 2]).unwrap();
        assert!(trie.contains(&[1, 2]));
    }

    #[test]
    fn test_contains_wrong_length() {
        let trie = CodeTrie::new(3);
        assert!(!trie.contains(&[0, 1]));
    }

    #[test]
    fn test_search_top_k() {
        let (trie, scores) = make_trie_and_scores();
        let result = trie.search(&scores, 2).unwrap();
        assert_eq!(result, vec![vec![0, 2], vec![1, 1]]);
    }

    #[test]
    fn test_search_all_results() {
        let (trie, scores) = make_trie_and_scores();
        let result = trie.search(&scores, 10).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![0, 2]);
        assert_eq!(result[1], vec![1, 1]);
        assert_eq!(result[2], vec![0, 0]);
    }

    #[test]
    fn test_search_k_one() {
        let (trie, scores) = make_trie_and_scores();
        let result = trie.search(&scores, 1).unwrap();
        assert_eq!(result, vec![vec![0, 2]]);
    }

    #[test]
    fn test_search_k_zero() {
        let (trie, scores) = make_trie_and_scores();
        let result = trie.search(&scores, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_search_empty_trie() {
        let trie = CodeTrie::new(2);
        let scores = ScoredBooks::new(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 2, 3);
        let result = trie.search(&scores, 3).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_search_depth_mismatch() {
        let trie = CodeTrie::new(2);
        let scores = ScoredBooks::new(vec![1.0; 9], 3, 3);
        assert!(matches!(
            trie.search(&scores, 1),
            Err(TrieError::BookNumberMismatch(3, 2))
        ));
    }
}
