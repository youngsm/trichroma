"""Extract a segment of explicit-contraction PTX as a Triton inline-asm block.

Development tool used to generate the long straight-line blocks of
:mod:`trichroma.engine.exact` from the original kernel's PTX after all
ptxas contraction decisions were made explicit (see
``docs/design.md``). Loads inside the segment become inputs,
stores are dropped, registers read before being written become inputs, and
the requested registers become outputs.
"""

import re
import sys

REG = re.compile(r'%(?:fd|rd|rs|fo|fn|fx|fm|f|r|p)\d+')


def reg_type(name):
    if name.startswith(('%fd',)):
        return 'f64'
    if name.startswith(('%rd',)):
        return 'b64'
    if name.startswith(('%rs',)):
        return 'b16'
    if name.startswith(('%f',)):
        return 'f32'
    if name.startswith('%p'):
        return 'pred'
    if name.startswith('%r'):
        return 'b32'
    raise ValueError(name)


def extract(lines, outputs, rename_labels=True, drop_ops=('st.',), input_order=None, prefix='cx'):
    """Returns (asm, constraints, inputs).

    Every register of the block is renamed ``%<prefix><name>`` (``%f493`` ->
    ``%cxf493``). The block's registers are declared inside ``{ }`` but the
    ``$n`` operands are the *enclosing* kernel's registers, which LLVM also
    names ``%f<N>``/``%r<N>``/``%p<N>``: without the prefix a scoped
    declaration can shadow the very register an operand refers to.
    """
    body = []
    defined = set()
    inputs = []
    used = set()
    decl = {}
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith('//'):
            continue
        m = re.match(r'\.reg\s+\.(\w+)\s+(%\w+);', s)
        if m:
            decl[m.group(2)] = m.group(1)
            continue
        if s.endswith(':'):
            body.append(('label', s[:-1]))
            continue
        text = s.split('//')[0].strip()
        pred = None
        t = text
        if t.startswith('@'):
            pred, t = t.split(None, 1)
        op = t.split()[0]
        args = t[len(op):].strip().rstrip(';')
        parts = [a.strip() for a in args.split(',')] if args else []
        if op.startswith(drop_ops):
            continue
        if op.startswith('ld.'):
            dst = parts[0]
            if dst not in defined and dst not in inputs:
                inputs.append(dst)
            defined.add(dst)
            continue
        srcs = parts[1:] if not op.startswith(('bra',)) else []
        if op.startswith('bra'):
            body.append(('bra', pred, text))  # keep the guard: '@%p bra L;'
            if pred:
                r = pred.lstrip('@!')
                if r not in defined and r not in inputs:
                    inputs.append(r)
            continue
        for a in srcs:
            for r in REG.findall(a):
                if r not in defined and r not in inputs:
                    inputs.append(r)
        if pred:
            r = pred.lstrip('@!')
            if r not in defined and r not in inputs:
                inputs.append(r)
        if parts:
            for r in REG.findall(parts[0]):
                defined.add(r)
        body.append(('ins', text))
    if input_order is not None:
        missing = [r for r in inputs if r not in input_order]
        if missing:
            raise ValueError('unlisted inputs: %s' % missing)
        inputs = [r for r in input_order if r in inputs] + [r for r in input_order if r not in inputs]
    regs = set()
    for kind, *rest in body:
        text = rest[-1] if kind != 'label' else ''
        regs.update(REG.findall(text))
        if kind == 'bra' and rest[0]:
            regs.add(rest[0].lstrip('@!'))
    regs.update(inputs)
    regs.update(outputs)
    def rn(text):
        return REG.sub(lambda m: '%' + prefix + m.group(0)[1:], text)

    labels = {}
    out = ['{']
    for r in sorted(regs):
        out.append('.reg .%s %s;' % (reg_type(r), rn(r)))
    nout = len(outputs)
    for i, r in enumerate(inputs):
        typ = reg_type(r)
        if typ == 'pred':
            out.append('setp.ne.u32 %s, $%d, 0;' % (rn(r), nout + i))
        elif typ == 'f32':
            out.append('mov.f32 %s, $%d;' % (rn(r), nout + i))
        elif typ == 'b32':
            out.append('mov.b32 %s, $%d;' % (rn(r), nout + i))
        elif typ == 'b64':
            out.append('mov.b64 %s, $%d;' % (rn(r), nout + i))
        else:
            raise ValueError('unsupported input type %s' % r)
    for kind, *rest in body:
        if kind == 'label':
            out.append('%s:' % rest[0].replace('$', 'X_'))
        else:
            out.append(rn(rest[-1]).replace('$L__', 'X_L__') + ('' if rest[-1].endswith(';') else ';'))
    for i, r in enumerate(outputs):
        typ = reg_type(r)
        if typ == 'f32':
            out.append('mov.f32 $%d, %s;' % (i, rn(r)))
        elif typ == 'b32':
            out.append('mov.b32 $%d, %s;' % (i, rn(r)))
        elif typ == 'pred':
            out.append('selp.u32 $%d, 1, 0, %s;' % (i, rn(r)))
        else:
            raise ValueError(r)
    out.append('}')
    cons = []
    for r in outputs:
        cons.append('=f' if reg_type(r) == 'f32' else '=r')
    for r in inputs:
        t = reg_type(r)
        cons.append('f' if t == 'f32' else ('l' if t == 'b64' else 'r'))
    return '\n'.join(out), ','.join(cons), inputs


if __name__ == '__main__':
    path, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    outs = sys.argv[4].split(',')
    lines = open(path).read().splitlines()[a - 1:b]
    asm, cons, ins = extract(lines, outs)
    print(asm)
    print(cons)
    print(ins)
