"""Lower the selected original FP32 wire primitive to inline Triton PTX.

Pointer addressing uses a contiguous immutable descriptor table. The quadratic
coefficient explicitly retains the fused operation order emitted inside the
original transport kernel; NVCC chooses a different order for an isolated query.
This is a legacy compiler adapter, not an independent wire algorithm. Its output
must be checked against unmodified original photon trajectories before use.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference_root.resolve()
    header_path = reference / "chroma/cuda/photon.h"
    header = header_path.read_text()
    begin = header.index("    int nplanes = g->nwireplanes;", header.index("fill_state(State"))
    end = header.index("    Material *material1 = 0;", begin)
    source = header[begin:end]
    if "wp->u_norm" not in source:
        raise ValueError("this adapter targets the installed FP32 wire primitive")
    source = source.replace("int nplanes = g->nwireplanes;", "int nplanes = plane_count;")
    source = source.replace("g->wireplanes != 0", "planes != 0")
    source = source.replace("g->wireplanes[ip]", "planes + ip")
    quadratic = "float A = dv*dv + dn*dn;"
    if source.count(quadratic) != 1:
        raise ValueError("unexpected original wire quadratic source")
    # The audited integrated original kernel emits mul(dv,dv), fma(dn,dn,mul).
    # Isolating the identical C++ expression reverses the rounded product and
    # changes near-tangent roots. Intrinsics preserve the historical lowering.
    source = source.replace(quadratic, "float A = __fmaf_rn(dn, dn, __fmul_rn(dv, dv));")
    wrapper = (
        r"""
#include "photon.h"
extern "C" __global__ void wire_probe(
    const float3 *positions, const float3 *directions, const WirePlane *planes,
    float *distance, float3 *normal, int *material_from, int *material_to,
    int *surface, int *triangle, int plane_count, int row, int valid)
{
    if (!valid) return;
    Photon p;
    p.position = positions[row];
    p.direction = directions[row];
    float best_distance = triangle[row] == -1 ? 1e30f : distance[row];
"""
        + source
        + r"""
    if (use_analytic) {
        distance[row] = analytic_distance;
        bool outside = analytic_dot_raw > 0.0f;
        normal[row] = outside ? analytic_normal_raw : -analytic_normal_raw;
        material_from[row] = outside ? analytic_mat_outer : analytic_mat_inner;
        material_to[row] = outside ? analytic_mat_inner : analytic_mat_outer;
        surface[row] = analytic_surface;
        triangle[row] = -2;
    }
}
"""
    )
    wrapper = "\n".join(line.rstrip() for line in wrapper.splitlines()) + "\n"
    sys.path.insert(0, str(reference))
    from chroma.cuda import srcdir
    from chroma.gpu.tools import cuda_options
    from pycuda.compiler import compile

    if Path(srcdir).resolve() != reference / "chroma/cuda":
        raise ValueError("the requested reference CUDA headers were not imported")
    ptx = compile(
        wrapper,
        no_extern_c=True,
        target="ptx",
        arch="sm_80",
        cache_dir=False,
        options=[*cuda_options, "-I" + str(srcdir)],
    ).decode()
    entry = re.search(r"\.visible \.entry wire_probe\((.*?)\)\s*\{", ptx, re.S)
    if entry is None:
        raise ValueError("NVCC did not emit the expected scalar primitive")
    depth, stop = 1, entry.end()
    while depth:
        depth += (ptx[stop] == "{") - (ptx[stop] == "}")
        stop += 1
    body = ptx[entry.end() : stop - 1].replace("$", "WIRE_")
    # LLVM substitutes operands using its own %r/%rd names. A nested PTX
    # declaration with those same names would shadow the substituted inputs.
    # Give every register owned by the adapter a separate namespace first.
    body = re.sub(r"%([A-Za-z][A-Za-z0-9_]*)", r"%wire_\1", body)
    params = re.findall(r"\.param\s+\.(\w+)\s+(wire_probe_param_\d+)", entry.group(1))
    if len(params) != 12:
        raise ValueError(f"unexpected wire primitive ABI: {params}")
    for index, (dtype, name) in enumerate(params):
        pattern = rf"ld\.param\.{dtype}\s+([^,]+),\s*\[{name}\];"
        body, count = re.subn(pattern, rf"mov.{dtype} \1, ${index+1};", body)
        if count < 1:
            raise ValueError(f"unexpected parameter load for {name}: {count}")
    if "ld.param" in body or "call" in body or "%tid" in body or "%ctaid" in body:
        raise ValueError("the lowered primitive is not self-contained scalar PTX")
    body = re.sub(r"\bret;", "bra LEGACY_WIRE_DONE;", body)
    asm = "{\n" + body + "\nLEGACY_WIRE_DONE:\nmov.u32 $0, 0;\n}\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        '"""Generated legacy FP32 wire primitive; see adjacent provenance JSON."""\n\nASM = r"""'
        + asm
        + '"""\n'
    )
    record = {
        "reference": str(reference),
        "reference_photon_header_sha256": hashlib.sha256(header_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(wrapper.encode()).hexdigest(),
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "adapter_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "cuda_options": list(cuda_options),
        "architecture": "sm_80",
        "arithmetic_contract": "quadratic A = fma.rn.ftz(dn,dn,mul.rn.ftz(dv,dv)), matching the audited integrated original transport lowering",
        "scope": "NVCC lowering of the original numerical wire primitive, scheduled inside Triton; not a CUDA transport-kernel dispatch",
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    args.output.with_suffix(".cu").write_text(wrapper)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
