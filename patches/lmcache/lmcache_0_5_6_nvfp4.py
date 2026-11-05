#!/usr/bin/env python3
"""Correctifs LMCache 0.5.6 pour un KV cache NVFP4 (pages de 72 octets, dtype uint8).

A appliquer AVANT le demarrage du serveur LMCache MP (les transferts y tournent).

1. D2H (store) : le fallback Python lance un cudaMemcpy unique qui franchit une
   frontiere de cudaHostRegister (PIN_CHUNK_SIZE = 64 Mo) des qu'un objet KV est
   grand (256k tokens = 37 Mo en NVFP4) -> `cudaMemcpy failed with error code 1`,
   store perdu, puis `assert req_id in self.requests` -> EngineDeadError.
   On decoupe comme le fait la version C++.

2. H2D (retrieve) : chaque scatter ecrit un staging d'un dtype different dans la
   page KV uint8 -> `index_copy_(): ... got (self) Byte and (source)
   Float8_e4m3fn`. Les 72 octets par token sont opaques : on reinterprete les
   octets des que la taille d'element est identique (no-op sinon).

Tous les `.index_copy_(0, idx, src)` sont reecrits en `_nvfp4_scatter(dst, idx, src)`.

Usage : patch_lmcache_nvfp4.py [racine_lmcache]
Idempotent ; echec bruyant si une ancre a change.
"""
import pathlib
import sys

ROOT = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/lmcache"
)
MK = ROOT / "v1/platform/torch_ops/mem_kernels.py"
MP = ROOT / "v1/platform/torch_ops/mp_mem_kernels.py"

OLD_COPY = """    libcudart = _get_copy_lib()
    if libcudart is not None and hasattr(libcudart, "cudaMemcpy"):
        try:
            # Synchronous cudaMemcpy handles cross-cudaHostRegister boundaries
            # internally — no manual alignment splitting needed.
            ret = libcudart.cudaMemcpy(
                ctypes.c_void_p(dest),
                ctypes.c_void_p(src),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(4),  # cudaMemcpyDefault
            )
            if ret != 0:
                raise RuntimeError(f"cudaMemcpy failed with error code {ret}")
        except AttributeError:
            raise
"""

NEW_COPY = """    libcudart = _get_copy_lib()
    if libcudart is not None and hasattr(libcudart, "cudaMemcpy"):
        try:
            # NVFP4CHUNK: never span a cudaHostRegister boundary (pinned chunk).
            _base_off = int(host_buffer_offset) if host_buffer_offset else 0
            _align = int(host_buffer_alignments) if host_buffer_alignments else 0
            remaining = int(nbytes)
            done = 0
            while remaining > 0:
                piece = remaining
                if _align > 0:
                    piece = min(piece, _align - ((_base_off + done) % _align))
                ret = libcudart.cudaMemcpy(
                    ctypes.c_void_p(dest + done),
                    ctypes.c_void_p(src + done),
                    ctypes.c_size_t(piece),
                    ctypes.c_int(4),  # cudaMemcpyDefault
                )
                if ret != 0:
                    raise RuntimeError(
                        "cudaMemcpy failed with error code %d (offset=%d piece=%d nbytes=%d)"
                        % (ret, done, piece, nbytes)
                    )
                done += piece
                remaining -= piece
        except AttributeError:
            raise
"""

HELPER = '''

def _nvfp4_scatter(dst, dim, index, src):
    """NVFP4DTYPE: scatter `src` into `dst` reinterpreting 1-byte dtypes.

    An NVFP4 KV page is uint8 (72 bytes/token) and those bytes are opaque: a real
    dtype conversion would corrupt them, so we only ever *view*.
    """
    if src.dtype != dst.dtype and src.element_size() == dst.element_size():
        src = src.view(dst.dtype)
    dst.index_copy_(dim, index, src)

'''

MARKER = ".index_copy_("


def _receiver_start(src: str, dot_pos: int) -> int:
    """Start index of the expression `expr.index_copy_(` at dot_pos."""
    k = dot_pos
    depth = 0
    while k > 0:
        ch = src[k - 1]
        if ch in ")]":
            depth += 1
        elif ch in "([":
            depth -= 1
            if depth < 0:
                break
        elif ch.isalnum() or ch in "_.[]":
            pass
        elif depth > 0 and ch in " ,:\t":
            pass
        else:
            break
        k -= 1
    return k


def _wrap_scatter(src: str) -> tuple[str, int]:
    out, i, n = [], 0, 0
    while True:
        j = src.find(MARKER, i)
        if j < 0:
            out.append(src[i:])
            break
        k = _receiver_start(src, j)
        recv = src[k:j]
        open_paren = j + len(MARKER) - 1
        depth, m = 0, open_paren
        while m < len(src):
            if src[m] == "(":
                depth += 1
            elif src[m] == ")":
                depth -= 1
                if depth == 0:
                    break
            m += 1
        inner = src[open_paren + 1 : m]
        if "_nvfp4_scatter(" not in inner and inner.lstrip().startswith("0,"):
            out.append(src[i:k] + "_nvfp4_scatter(" + recv + ", " + inner + ")")
            n += 1
        else:
            out.append(src[i : m + 1])
        i = m + 1
    return "".join(out), n


def _read(path: pathlib.Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _write(path: pathlib.Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def patch_mem_kernels() -> None:
    src = _read(MK)
    if "NVFP4CHUNK" in src:
        print("D2H chunk-split: deja applique")
        return
    if OLD_COPY not in src:
        print("D2H chunk-split: ANCRE INTROUVABLE")
        sys.exit(2)
    _write(MK, src.replace(OLD_COPY, NEW_COPY, 1))
    print("D2H chunk-split: applique")


def patch_mp_kernels() -> None:
    src = _read(MP)
    if "NVFP4DTYPE" in src:
        print("H2D dtype-bridge: deja applique")
        return
    src = src.replace(
        "                    obj[kv, layer_idx, offset_in_object:token_end].copy_(\n"
        "                        flat, non_blocking=True\n                    )",
        "                    _dst = obj[kv, layer_idx, offset_in_object:token_end]\n"
        "                    if flat.dtype != _dst.dtype and flat.element_size() == _dst.element_size():\n"
        "                        flat = flat.view(_dst.dtype)\n"
        "                    _dst.copy_(flat, non_blocking=True)",
    )
    src, n = _wrap_scatter(src)
    anchor = "\ndef _transfer_cross_layer("
    if anchor not in src:
        print("H2D dtype-bridge: ANCRE helper introuvable")
        sys.exit(2)
    src = src.replace(anchor, HELPER + anchor, 1)
    _write(MP, src)
    print(f"H2D dtype-bridge: {n} scatter enveloppes")


patch_mem_kernels()
patch_mp_kernels()
print("LMCACHE_NVFP4_PATCHES_READY")
