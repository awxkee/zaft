/*
 * // Copyright (c) Radzivon Bartoshyk 1/2026. All rights reserved.
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
use crate::{R2CFftExecutor, ZaftError};
use std::sync::Arc;

pub(crate) trait R2cAlgorithmFactory<T> {
    fn r2c_butterfly1() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly2() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly3() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly4() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly5() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly6() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly7() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly8() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly9() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly11() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly12() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly13() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly14() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly15() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly16() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly17() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly19() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly23() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly29() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly31() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_butterfly32() -> Arc<dyn R2CFftExecutor<T> + Send + Sync>;
    fn r2c_raders(n: usize) -> Result<Arc<dyn R2CFftExecutor<T> + Send + Sync>, ZaftError>;
    fn r2c_bluestein(n: usize) -> Result<Arc<dyn R2CFftExecutor<T> + Send + Sync>, ZaftError>;
}
