# Example: two warp roles, one CTA

`two_roles.cu` is a producer warp and a consumer warp handing a shared-memory tile back and forth over two named
barriers, in a window loop. The stamps are switched by the file's own `constexpr` conditions (`Coarse`: the launch's
two edges and each phase's start, as marks; `Detail`: every wait and arrival, as stamps). The host side copies the rows
back and writes a warpline record through `include/warpline/host.hpp`.

Build (nvcc, CUDA 11+; change `sm_80` to your GPU's architecture; no other dependency):

    nvcc -std=c++17 -arch=sm_80 -I../include two_roles.cu -o two_roles

Run, and read the record:

    ./two_roles record.json
    python -m warpline check  record.json      # from the repository root, or with warpline installed
    python -m warpline window record.json      # the average window, each stamp's offset in it
    python -m warpline rates  record.json      # the cycle counter's rate against time, from the ties

The program also times the plain kernel (`<false, false>`), the coarse one and coarse + detail, and prints their
milliseconds.

See the compiled-out claim for yourself: the kernel with both switches off holds no clock read.

    cuobjdump -sass two_roles | awk '/Function : .*ILb0ELb0/{f=1} /Function : .*ILb[01]ELb[01]/&&!/ILb0ELb0/{f=0} f' \
      | grep -cE 'CLOCK|GLOBALTIMER'        # 0

(The instantiation names are `ILb<Coarse>ELb<Detail>`.)

## What a site compiles to (`cuobjdump -sass`, sm_86)

| site | instructions |
|---|---|
| `stamp` at a caller-held address | `CS2R` clock, `MOV` id, `ST.E.64`: 3 |
| `stamp_if` (one-lane predicate) | `ISETP`, `CS2R`, `@P MOV`, `@P ST.E.64`: 4 |
| `Cursor::stamp` (capacity check, address) | `ISETP`, `IMAD.WIDE`, `CS2R`, `@P MOV`, `@P ST.E.64`: 5 (plus an `IADD3` where the running count is not folded) |
| `Cursor::mark` | `ISETP`, `IMAD.WIDE`, 2 `CS2R` (clock, timer), 2 `@P MOV`, 2 `@P ST.E.64`: 8 |
| a wait-end stamp after `bar.sync` | the stamp's 5, plus `MEMBAR.SC.CTA` from `after_wait()` |

ptxas may put a branch around a mark's two predicated stores where the mark is outside a loop (it did for the example's
launch-entry mark); inside the window loop they are predicated.
