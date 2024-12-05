# %%
# type: ignore
import torch as t
import triton.language as tl
import triton

@triton.jit
def sequential_scan_kernel(
    x_ptr,
    out_ptr,
    block_sum_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x
    for i in range(BLOCK_SIZE):
        curv_mask = tl.arange(0, BLOCK_SIZE) == i-1
        curv = tl.sum(tl.where(curv_mask, x, 0))
        add_to_mask = tl.arange(0, BLOCK_SIZE) >= i
        out = tl.where(add_to_mask, out + curv, out)

    last_elem_mask = tl.arange(0, BLOCK_SIZE) == (BLOCK_SIZE - 1)
    block_sum = tl.sum(tl.where(last_elem_mask & mask, out, 0))
    tl.store(block_sum_ptr + pid, block_sum)
    tl.store(out_ptr + offsets, out, mask=mask)


# @triton.jit
# def

def scan(x: t.Tensor):
    out = t.empty_like(x).cuda()
    n = out.numel()
    BLOCK_SIZE = 4
    n_blocks = triton.cdiv(n, BLOCK_SIZE)
    block_sums = t.empty(n_blocks, dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    sequential_scan_kernel[grid](x, out, block_sums, n, BLOCK_SIZE=BLOCK_SIZE)

    assert n_blocks < BLOCK_SIZE
    cum_block_sums = t.empty_like(block_sums).cuda()
    garbage = t.empty(n_blocks).cuda()
    sequential_scan_kernel[grid](block_sums, cum_block_sums, garbage, n_blocks, BLOCK_SIZE=BLOCK_SIZE)
    print(block_sums)
    print(cum_block_sums)


    return out

# inp = t.randint(0, 10000, size=(100,)).cuda()
inp = t.arange(10).cuda()
outp = scan(inp)
print("Your result:", outp)
print("Expected:", t.cumsum(inp, 0))
t.testing.assert_close(outp, t.cumsum(inp, 0))
# %%
