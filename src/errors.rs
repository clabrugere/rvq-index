#[cfg(feature = "safetensors")]
use safetensors::SafeTensorError;
use thiserror::Error;

use std::io::Error as IoError;

use crate::codebook::Code;

pub type CodeBooksResult<R> = Result<R, CodeBooksError>;
pub type TrieResult<R> = Result<R, TrieError>;
pub type RvqIndexResult<R> = Result<R, RvqIndexError>;

#[derive(Debug, Error)]
pub enum CodeBooksError {
    #[error("Inconsistent input shapes, got {0} while expected {1}")]
    InconsistentShapes(usize, usize),
    #[error("Query dimension mismatch, got {0} while expected {1}")]
    QueryDimensionMismatch(usize, usize),
    #[error("IO error: {0}")]
    Io(#[from] IoError),
    #[error("Expected 3D array [num_books, num_codes, dim], got {0}D")]
    FileInconsistentShape(usize),
    #[cfg(feature = "safetensors")]
    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] SafeTensorError),
}

#[derive(Debug, Error)]
pub enum TrieError {
    #[error("Code length must match index depth, got {0} while expected {1}")]
    CodesLengthMismatch(usize, usize),
    #[error("{0} not found")]
    NotFound(Code),
    #[error("Number of books in ScoredBooks should match depth, got {0} while expected {1}")]
    BookNumberMismatch(usize, usize),
}

#[derive(Debug, Error)]
pub enum RvqIndexError {
    #[error("Invalid code in input, code should be less than number of codes in codebooks")]
    InvalidCode,
    #[error("Trie error: {0}")]
    TrieError(#[from] TrieError),
    #[error("CodeBook error: {0}")]
    CodeBooksError(#[from] CodeBooksError),
}
