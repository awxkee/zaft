# zaft — FFT for JavaScript / TypeScript

[![npm](https://img.shields.io/npm/v/zaft)](https://www.npmjs.com/package/@radzivon.bartoshyk/zaft)
[![CI](https://github.com/awxkee/zaft/actions/workflows/npm_publish.yml/badge.svg)]([https://github.com/awxkee/zaft/.github/workflows/npm_publish.yml](https://github.com/awxkee/zaft/blob/master/.github/workflows/npm_publish.yml))

High-performance FFT for the browser and Node.js, compiled from Rust to WebAssembly with **SIMD acceleration** (`simd128`).

## Installation

```bash
yarn add @radzivon.bartoshyk/zaft
```

## Complex number layout

All complex transforms use **interleaved typed arrays**:

```
[re₀, im₀, re₁, im₁, …]   length = 2 × n
```

This is the fastest layout for WASM ↔ JS data transfer and is compatible with the Web Audio API's `AnalyserNode` pattern.

## Quick start

### Stateless functions

```ts
import { fft, ifft, rfft, irfft, fftfreq } from 'zaft';

// Build a complex signal (interleaved: [re, im, re, im, ...])
const n = 1024;
const signal = new Float64Array(n * 2);
for (let i = 0; i < n; i++) signal[i * 2] = Math.sin(2 * Math.PI * 440 * i / 44100);

// Forward FFT
const spectrum = fft(signal);                    // Float64Array, length 2*n
const freqs    = fftfreq(n, 1 / 44100);         // Float64Array, length n

// Inverse FFT
const recovered = ifft(spectrum);                // Float64Array, length 2*n
```

### Real transforms (half the output)

```ts
import { rfft, irfft, rfftfreq } from 'zaft';

const real = new Float64Array(1024);
// ... fill with real samples ...

const spectrum = rfft(real);                     // Float64Array, length 2*(n/2+1)
const freqs    = rfftfreq(1024, 1 / 44100);

const back = irfft(spectrum, { n: 1024 });       // Float64Array, length n
```

### Planner — best for repeated transforms

```ts
import { Plan, Plan32 } from 'zaft';

// Double precision
const plan = new Plan(1024);
const X = plan.forward(signal);
const x = plan.inverse(X);
// plan is freed automatically when GC'd — no manual .free() needed

// Single precision (half the memory, slightly lower accuracy)
const plan32 = new Plan32(1024);
const X32 = plan32.forward(new Float32Array(signal));
```

### Normalisation

```ts
// Same three modes as numpy/scipy:
fft(signal, { norm: 'backward' });  // default: no scale on forward
fft(signal, { norm: 'ortho'    });  // 1/√n on both directions
fft(signal, { norm: 'forward'  });  // 1/n on forward
```

### next_fast_len

```ts
import { next_fast_len } from 'zaft';
const n = next_fast_len(1000);   // → 1000 (13-smooth, efficient)
const m = next_fast_len(1001);   // → 1008
```

## API reference

| Function            | Input                      | Output                     | Notes                     |
|---------------------|----------------------------|----------------------------|---------------------------|
| `fft(a, opts?)`     | `Float64Array` interleaved | `Float64Array` interleaved | C2C forward               |
| `ifft(a, opts?)`    | `Float64Array` interleaved | `Float64Array` interleaved | C2C inverse               |
| `rfft(a, opts?)`    | `Float64Array` real        | `Float64Array` interleaved | R2C, length `2*(n/2+1)`   |
| `irfft(a, opts?)`   | `Float64Array` interleaved | `Float64Array` real        | C2R                       |
| `fft32(a, opts?)`   | `Float32Array` interleaved | `Float32Array` interleaved | f32 C2C forward           |
| `ifft32(a, opts?)`  | `Float32Array` interleaved | `Float32Array` interleaved | f32 C2C inverse           |
| `rfft32(a, opts?)`  | `Float32Array` real        | `Float32Array` interleaved | f32 R2C                   |
| `irfft32(a, opts?)` | `Float32Array` interleaved | `Float32Array` real        | f32 C2R                   |
| `fftfreq(n, d?)`    | —                          | `Float64Array`             | DFT sample frequencies    |
| `rfftfreq(n, d?)`   | —                          | `Float64Array`             | rfft sample frequencies   |
| `next_fast_len(n)`  | —                          | `number`                   | Nearest 13-smooth integer |
| `new Plan(n)`       | —                          | `Plan`                     | Pre-planned f64 executor  |
| `new Plan32(n)`     | —                          | `Plan32`                   | Pre-planned f32 executor  |

`opts` shape: `{ n?: number, norm?: 'backward' | 'ortho' | 'forward' }`

## Bundle targets

Three builds are produced — pick the one that matches your toolchain:

| Target    | Path             | Use when                             |
|-----------|------------------|--------------------------------------|
| `bundler` | `zaft` (default) | webpack / vite / rollup              |
| `web`     | `zaft/web`       | `<script type="module">`, no bundler |
| `nodejs`  | `zaft/node`      | Node.js / CommonJS                   |

## License

BSD 3-Clause