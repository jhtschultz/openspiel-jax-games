"""Simple JAX GPU benchmark: batched matrix multiply."""

import jax
import jax.numpy as jnp
import time

def benchmark_matmul(batch_size=64, matrix_size=1024, num_runs=100):
    """Benchmark batched matrix multiplication."""

    print(f"JAX devices: {jax.devices()}")
    print(f"\nBenchmark: {batch_size}x ({matrix_size}x{matrix_size}) @ ({matrix_size}x{matrix_size})")
    print("-" * 50)

    # Create random matrices
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (batch_size, matrix_size, matrix_size))
    B = jax.random.normal(k2, (batch_size, matrix_size, matrix_size))

    # JIT compile the matmul
    @jax.jit
    def batched_matmul(a, b):
        return jnp.matmul(a, b)

    # Warmup (includes compilation)
    print("Warming up (JIT compile)...")
    result = batched_matmul(A, B)
    result.block_until_ready()

    # Benchmark
    print(f"Running {num_runs} iterations...")
    start = time.perf_counter()
    for _ in range(num_runs):
        result = batched_matmul(A, B)
        result.block_until_ready()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_runs) * 1000
    total_flops = batch_size * (2 * matrix_size**3) * num_runs
    tflops = total_flops / elapsed / 1e12

    print(f"\nResults:")
    print(f"  Average time: {avg_ms:.2f} ms")
    print(f"  Throughput:   {tflops:.2f} TFLOPS")
    print(f"  Output shape: {result.shape}")

if __name__ == "__main__":
    benchmark_matmul()
