use thiserror::Error;

use super::codebook::Code;

pub type ScoredCodeBookResult<R> = Result<R, ScoredCodeBookError>;
pub type CodeBooksResult<R> = Result<R, CodeBooksError>;
pub type TrieResult<R> = Result<R, TrieError>;
pub type RvqIndexResult<R> = Result<R, RvqIndexError>;

#[derive(Debug, Error)]
pub enum ScoredCodeBookError {
    #[error("Inconsistent input shapes, got {0} while expected {1}")]
    InconsistentShapes(usize, usize),
}

#[derive(Debug, Error)]
pub enum CodeBooksError {
    #[error("Inconsistent input shapes, got {0} while expected {1}")]
    InconsistentShapes(usize, usize),
    #[error("Query dimension mismatch")]
    QueryDimensionMismatch,
    #[error("ScoredCodeBook error: {0}")]
    ScoredCodeBookError(#[from] ScoredCodeBookError),
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
    #[error("Trie error: {0}")]
    TrieError(#[from] TrieError),
    #[error("CodeBook error: {0}")]
    CodeBooksError(#[from] CodeBooksError),
}
