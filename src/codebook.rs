use core::{
    cmp::{Ordering, PartialOrd},
    iter::Sum,
    marker::{Copy, Send, Sync},
    ops::{Add, Mul},
};
#[cfg(any(feature = "npy", feature = "safetensors"))]
use std::path::Path;

#[cfg(feature = "npy")]
use {
    npyz::{Deserialize as NpyDeserialize, NpyFile},
    std::{fs::File, io::BufReader},
};

#[cfg(feature = "safetensors")]
use {bytemuck::Pod, safetensors::SafeTensors};

use crate::errors::{CodeBooksError, CodeBooksResult};

// Bounded traits for embedding types with blanket implementation
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
#[inline(always)]
fn dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    assert_eq!(x.len(), y.len());
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

    pub fn get_book_max(&self, book: usize) -> T
    where
        T: Scalar,
    {
        self.get_book(book)
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or_default()
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
                data.len(),
                num_books * num_codes * dim,
            ));
        }

        Ok(Self {
            data,
            num_books,
            num_codes,
            dim,
        })
    }

    pub fn score(&self, query: &[T]) -> CodeBooksResult<ScoredBooks<T>>
    where
        T: Scalar,
    {
        if query.len() != self.dim {
            return Err(CodeBooksError::QueryDimensionMismatch(
                query.len(),
                self.dim,
            ));
        }

        let scores = self
            .data
            .chunks_exact(self.dim)
            .map(|embedding| dot(query, embedding))
            .collect();

        Ok(ScoredBooks::new(scores, self.num_books, self.num_codes))
    }
}

// Optionally allow to load code books from safetensors
#[cfg(feature = "safetensors")]
impl<T: Scalar + Pod> CodeBooks<T> {
    pub fn from_safetensors(path: impl AsRef<Path>, tensor_name: &str) -> CodeBooksResult<Self> {
        let bytes = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let view = tensors.tensor(tensor_name)?;
        let shape = view.shape();

        if shape.len() != 3 {
            return Err(CodeBooksError::FileInconsistentShape(shape.len()));
        }

        let (num_books, num_codes, dim) = (shape[0], shape[1], shape[2]);
        let data = bytemuck::cast_slice::<u8, T>(view.data()).to_vec();
        Self::new(data, num_books, num_codes, dim)
    }
}

// Optionally allow to load code books from npy files
#[cfg(feature = "npy")]
impl<T: Scalar + NpyDeserialize> CodeBooks<T> {
    pub fn from_npy(path: impl AsRef<Path>) -> CodeBooksResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let npy = NpyFile::new(reader)?;
        let shape = npy
            .shape()
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<usize>>();

        if shape.len() != 3 {
            return Err(CodeBooksError::FileInconsistentShape(shape.len()));
        }

        let (num_books, num_codes, dim) = (shape[0], shape[1], shape[2]);
        let data = npy.into_vec()?;
        Self::new(data, num_books, num_codes, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_codebooks() -> CodeBooks<f32> {
        let data: Vec<f32> = vec![
            1.0, 0.0, 0.0, // Book 1
            0.0, 1.0, 0.0, // Book 2
            0.0, 0.0, 1.0, // Book 3
        ];
        CodeBooks::new(data, 3, 1, 3).unwrap()
    }

    #[test]
    fn test_dot_f32() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0];
        let y: Vec<f32> = vec![4.0, 5.0, 6.0];
        assert_eq!(dot(&x, &y), 32.0);
    }

    #[test]
    fn test_creation_wrong_shapes() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0]; // Should be num_books * num_codes * dim
        let result = CodeBooks::new(data, 1, 1, 2); // Expecting 2 elements for dim
        assert!(matches!(
            result,
            Err(CodeBooksError::InconsistentShapes(3, 2))
        ));
    }

    #[test]
    fn test_score() {
        let codebooks = create_codebooks();
        let query: Vec<f32> = vec![1.0, 2.0, 3.0];
        let scores = codebooks.score(&query).unwrap();

        assert_eq!(scores.num_books, 3);
        assert_eq!(scores.num_codes, 1);
        assert_eq!(scores.get_book(0), &[1.0]);
        assert_eq!(scores.get_book(1), &[2.0]);
        assert_eq!(scores.get_book(2), &[3.0]);
    }

    #[test]
    fn test_score_dimension_mismatch() {
        let codebooks = create_codebooks();
        let query: Vec<f32> = vec![1.0, 2.0]; // Wrong dimension
        let result = codebooks.score(&query);
        assert!(matches!(
            result,
            Err(CodeBooksError::QueryDimensionMismatch(2, 3))
        ));
    }

    #[test]
    fn test_get_book_max() {
        let codebooks = create_codebooks();
        let query: Vec<f32> = vec![1.0, 2.0, 3.0];
        let scores = codebooks.score(&query).unwrap();

        assert_eq!(scores.get_book_max(0), 1.0);
        assert_eq!(scores.get_book_max(1), 2.0);
        assert_eq!(scores.get_book_max(2), 3.0);
    }
}
