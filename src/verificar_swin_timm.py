"""Compara SwinBranch con el Swin-T de timm usando los mismos pesos (requiere torch y timm)."""
import os, sys
os.environ["DF_BACKEND"] = "numpy"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, timm
import backend as bk
bk.set_dtype(np.float64); bk.set_seed(42); torch.manual_seed(42)
from swin import SwinBranch, SwinBlock, PatchMerging

ref = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0).double().eval()
for m in ref.modules():  # misma GELU (aprox. tanh) que la implementación desde cero
    if hasattr(m, "act") and isinstance(m.act, torch.nn.GELU):
        m.act = torch.nn.GELU(approximate="tanh")
with torch.no_grad():  # pesos aleatorios no triviales también en LN y sesgos
    for p in ref.parameters():
        p.add_(torch.randn_like(p) * 0.05)

mine = SwinBranch()
T = lambda t: t.detach().numpy().astype(np.float64)
def ln(dst, src): dst.gamma.data[...] = T(src.weight); dst.beta.data[...] = T(src.bias)
def lin(dst, src):
    dst.W.data[...] = T(src.weight).T
    if dst.b is not None: dst.b.data[...] = T(src.bias)

pe = ref.patch_embed
mine.patch_embed.proj.W.data[...] = T(pe.proj.weight).reshape(96, -1).T
mine.patch_embed.proj.b.data[...] = T(pe.proj.bias); ln(mine.patch_embed.norm, pe.norm)
ref_seq = []
for st in ref.layers:
    if not isinstance(st.downsample, torch.nn.Identity): ref_seq.append(st.downsample)
    ref_seq += list(st.blocks)
assert len(ref_seq) == len(mine.layers), (len(ref_seq), len(mine.layers))
for a, r in zip(mine.layers, ref_seq):
    if isinstance(a, PatchMerging):
        ln(a.norm, r.norm); lin(a.red, r.reduction)
    else:
        assert isinstance(a, SwinBlock)
        ln(a.norm1, r.norm1); ln(a.norm2, r.norm2)
        lin(a.attn.qkv, r.attn.qkv); lin(a.attn.proj, r.attn.proj)
        a.attn.bias_table.data[...] = T(r.attn.relative_position_bias_table)
        lin(a.mlp.mods[0], r.mlp.fc1); lin(a.mlp.mods[2], r.mlp.fc2)
        assert a.s == r.shift_size[0] if isinstance(r.shift_size, tuple) else a.s == r.shift_size, "desplazamiento distinto"
ln(mine.norm, ref.norm)

x = np.random.RandomState(0).randn(2, 3, 224, 224)
with torch.no_grad():
    y_ref = T(ref(torch.from_numpy(x)))  # (2, 768): norm + promedio global
y = mine.forward_features(bk.asarray(x))
err = np.abs(y - y_ref).max() / np.abs(y_ref).max()
print("shifts por bloque:", [l.s for l in mine.layers if isinstance(l, SwinBlock)])
print(f"salida (2,768): err. relativo máx. vs timm = {err:.2e}")
assert y.shape == y_ref.shape == (2, 768) and err < 1e-10
print("OK: forward idéntico a timm (máscara, sesgo relativo, roll y patch merging)")
