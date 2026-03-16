use core::{
    cmp::PartialOrd,
    iter::Sum,
    marker::{Copy, Send, Sync},
    ops::{Add, Mul},
};
use std::cmp::Ordering;

use super::errors::{CodeBooksError, CodeBooksResult, ScoredCodeBookError, ScoredCodeBookResult};

// Bounded traits to score embeddings with blanket implementation
pub trait Scalar:
    Default + Copy + PartialOrd + Send + Sync + Add<Output = Self> + Mul<Output = Self> + Sum
{
}
impl<T: Default + Copy + PartialOrd + Send + Sync + Add<Output = Self> + Mul<Output = Self> + Sum>
    Scalar for T
{
}

pub type Code = usize;

// Computes the dot product of two arrays of size D
fn dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum()
}

// Stores a code and its score with respect to a query
#[derive(Debug)]
pub struct ScoredCode<T> {
    pub code: Code,
    pub score: T,
}

impl<T: Scalar> ScoredCode<T> {
    pub fn new(code: Code, score: T) -> Self {
        Self { code, score }
    }
}

impl<T: Scalar> PartialEq for ScoredCode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.code == other.code
    }
}

impl<T: Scalar> Eq for ScoredCode<T> {}

impl<T: Scalar> PartialOrd for ScoredCode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Scalar> Ord for ScoredCode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then(self.code.cmp(&other.code))
    }
}

// Scores for all code books stored in a flat array
#[derive(Debug)]
pub struct ScoredBooks<T> {
    scores: Vec<ScoredCode<T>>,
    pub num_books: usize,
    pub num_codes: usize,
}

impl<T> ScoredBooks<T> {
    pub fn new(
        scores: Vec<ScoredCode<T>>,
        num_books: usize,
        num_codes: usize,
    ) -> ScoredCodeBookResult<Self> {
        if scores.len() != num_books * num_codes {
            return Err(ScoredCodeBookError::InconsistentShapes(
                num_books * num_codes,
                scores.len(),
            ));
        }

        Ok(Self {
            scores,
            num_books,
            num_codes,
        })
    }

    pub fn get_book(&self, book: usize) -> &[ScoredCode<T>] {
        let offset = book * self.num_codes;
        &self.scores[offset..offset + self.num_codes]
    }
}

// Stores embeddings in a flat array [L, K, D]
#[derive(Debug)]
pub struct CodeBooks<T> {
    data: Vec<T>,
    pub num_books: usize,
    pub num_codes: usize,
    pub dim: usize,
}

impl<T> CodeBooks<T> {
    pub fn new(
        data: Vec<T>,
        num_books: usize,
        num_codes: usize,
        dim: usize,
    ) -> CodeBooksResult<Self> {
        if data.len() != num_books * num_codes {
            return Err(CodeBooksError::InconsistentShapes(
                num_books * num_codes * dim,
                data.len(),
            ));
        }

        Ok(Self {
            data,
            num_books,
            num_codes,
            dim,
        })
    }

    pub fn get_embedding(&self, book: usize, code: usize) -> &[T] {
        let offset = (book * self.num_codes + code) * self.dim;
        &self.data[offset..offset + self.dim]
    }

    pub fn score(&self, query: &[T]) -> CodeBooksResult<ScoredBooks<T>>
    where
        T: Scalar,
    {
        if query.len() != self.dim {
            return Err(CodeBooksError::QueryDimensionMismatch);
        }

        let mut scores = Vec::with_capacity(self.num_books * self.num_codes);
        for book in 0..self.num_books {
            for code in 0..self.num_codes {
                let embedding = self.get_embedding(book, code);
                scores.push(ScoredCode::new(code, dot(query, embedding)));
            }
        }
        // sort codes descending by score within each codebook
        for codebook in scores.chunks_exact_mut(self.num_codes) {
            codebook.sort_unstable_by(|a, b| b.cmp(a));
        }
        let scored_books = ScoredBooks::new(scores, self.num_books, self.num_codes)?;

        Ok(scored_books)
    }
}
