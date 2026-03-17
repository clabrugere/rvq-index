use core::{
    cmp::PartialOrd,
    iter::Sum,
    marker::{Copy, Send, Sync},
    ops::{Add, Mul},
};

use super::errors::{CodeBooksError, CodeBooksResult};

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

// Scores for all code books stored in a flat array
#[derive(Debug)]
pub struct ScoredBooks<T> {
    scores: Vec<T>,
    pub num_books: usize,
    pub num_codes: usize,
}

impl<T> ScoredBooks<T> {
    pub fn new(scores: Vec<T>, num_books: usize, num_codes: usize) -> Self {
        Self {
            scores,
            num_books,
            num_codes,
        }
    }

    pub fn get_book(&self, book: usize) -> &[T] {
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
        if data.len() != num_books * num_codes * dim {
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

        let scores = self
            .data
            .chunks_exact(self.dim)
            .map(|embedding| dot(query, embedding))
            .collect();

        Ok(ScoredBooks::new(scores, self.num_books, self.num_codes))
    }
}
