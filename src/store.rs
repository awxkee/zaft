/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use std::ops::{Index, IndexMut, RangeFrom};

pub(crate) trait BidirectionalStore<T>:
    Index<usize, Output = T> + IndexMut<usize, Output = T> + Sized
{
    #[allow(unused)]
    fn slice_from(&self, range: RangeFrom<usize>) -> &[T];
    #[allow(unused)]
    /// Returns a mutable subslice starting from the given index
    fn slice_from_mut(&mut self, range: RangeFrom<usize>) -> &mut [T];
}

pub(crate) struct InPlaceStore<'a, T> {
    data: &'a mut [T],
}

impl<T> Index<usize> for InPlaceStore<'_, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.data.get_unchecked(index) }
    }
}

impl<T> IndexMut<usize> for InPlaceStore<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

impl<T> BidirectionalStore<T> for InPlaceStore<'_, T> {
    #[inline(always)]
    fn slice_from(&self, range: RangeFrom<usize>) -> &[T] {
        unsafe { self.data.get_unchecked(range.start..) }
    }

    #[inline(always)]
    fn slice_from_mut(&mut self, range: RangeFrom<usize>) -> &mut [T] {
        unsafe { self.data.get_unchecked_mut(range.start..) }
    }
}

impl<'a, T> InPlaceStore<'a, T> {
    pub(crate) fn new(data: &'a mut [T]) -> Self {
        Self { data }
    }
}

pub(crate) struct BiStore<'a, T> {
    read: &'a [T],
    write: &'a mut [T],
}

impl<T> Index<usize> for BiStore<'_, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.read.get_unchecked(index) }
    }
}

impl<T> IndexMut<usize> for BiStore<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.write.get_unchecked_mut(index) }
    }
}

impl<T> BidirectionalStore<T> for BiStore<'_, T> {
    #[inline(always)]
    fn slice_from(&self, range: RangeFrom<usize>) -> &[T] {
        unsafe { self.read.get_unchecked(range.start..) }
    }

    #[inline(always)]
    fn slice_from_mut(&mut self, range: RangeFrom<usize>) -> &mut [T] {
        unsafe { self.write.get_unchecked_mut(range.start..) }
    }
}

impl<'a, T> BiStore<'a, T> {
    pub(crate) fn new(read: &'a [T], write: &'a mut [T]) -> Self {
        Self { read, write }
    }
}
