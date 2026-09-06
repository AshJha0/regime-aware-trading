//! Minimal dense row-major matrix of `f64`.
//!
//! The project only needs plain storage with row access and left-to-right
//! reductions (the pinned summation order of API_SPEC.md §6), so a tiny
//! purpose-built type beats pulling in a linear-algebra dependency.

use crate::error::{RegimeError, Result};

/// Dense row-major `f64` matrix (`rows x cols`).
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// All-zero matrix of the given shape.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Matrix filled with a constant value.
    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        Matrix { rows, cols, data: vec![value; rows * cols] }
    }

    /// Build from a flat row-major vector; `data.len()` must equal `rows*cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(RegimeError::InvalidInput(format!(
                "matrix data length {} does not match shape {}x{}",
                data.len(),
                rows,
                cols
            )));
        }
        Ok(Matrix { rows, cols, data })
    }

    /// Build from row slices; every row must have the same length.
    pub fn from_rows(rows: &[Vec<f64>]) -> Result<Self> {
        let n = rows.len();
        let cols = rows.first().map_or(0, |r| r.len());
        let mut data = Vec::with_capacity(n * cols);
        for row in rows {
            if row.len() != cols {
                return Err(RegimeError::InvalidInput("ragged rows in matrix".into()));
            }
            data.extend_from_slice(row);
        }
        Ok(Matrix { rows: n, cols, data })
    }

    /// A `(T, 1)` column matrix from a 1-D series.
    pub fn column(series: &[f64]) -> Self {
        Matrix { rows: series.len(), cols: 1, data: series.to_vec() }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Element at `(r, c)`.  Panics on an out-of-range index like any slice
    /// access; every public library entry point validates shapes first, so
    /// this is only reachable through direct misuse of `Matrix` (use
    /// [`Matrix::try_get`] for a checked read).
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Set element at `(r, c)`.
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, value: f64) {
        self.data[r * self.cols + c] = value;
    }

    /// Row `r` as a slice.
    #[inline]
    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Mutable row `r`.
    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f64] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Copy of column `c`.
    pub fn col(&self, c: usize) -> Vec<f64> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    /// True iff every element is finite (no NaN/inf).
    pub fn all_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }

    /// Sub-matrix of the first `n` rows (used by expanding-window refits).
    ///
    /// # Errors
    /// `n > rows` (never a panic).
    pub fn head(&self, n: usize) -> Result<Matrix> {
        if n > self.rows {
            return Err(RegimeError::InvalidInput(format!(
                "head({n}) exceeds the matrix row count {}",
                self.rows
            )));
        }
        Ok(Matrix { rows: n, cols: self.cols, data: self.data[..n * self.cols].to_vec() })
    }

    /// Element at `(r, c)`, or an error when out of range (never a panic).
    pub fn try_get(&self, r: usize, c: usize) -> Result<f64> {
        if r >= self.rows || c >= self.cols {
            return Err(RegimeError::InvalidInput(format!(
                "index ({r}, {c}) out of range for a {}x{} matrix",
                self.rows, self.cols
            )));
        }
        Ok(self.data[r * self.cols + c])
    }
}
