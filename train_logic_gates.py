"""
Demo: MLP học các cổng logic AND, OR, NOT, XOR
=================================================
Minh họa lý thuyết quan trọng nhất của Deep Learning:
  1. Perceptron đơn (không có hidden layer) học được AND, OR, NOT
  2. Perceptron đơn KHÔNG học được XOR  ← điểm mấu chốt!
  3. MLP (có hidden layer) học được XOR

Lý do: XOR không phân tách tuyến tính → cần biểu diễn phi tuyến
       Đây chính là lý do cần hidden layers trong MLP.

Kết quả lưu tại: Models/logic_gates_demo/
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

torch.manual_seed(42)
np.random.seed(42)

OUT_DIR = 'Models/logic_gates_demo'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Truth tables ───────────────────────────────────────────────────────────────
GATES_2IN = {
    'AND': ([0,0,0,1], 'Phân tách tuyến tính ✓'),
    'OR':  ([0,1,1,1], 'Phân tách tuyến tính ✓'),
    'XOR': ([0,1,1,0], 'KHÔNG phân tách tuyến tính ✗'),
}
X2 = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])  # 4 điểm đầu vào 2 chiều

# NOT gate: 1 input
X1 = torch.tensor([[0.],[1.]])
Y_NOT = torch.tensor([1., 0.])

# ── Models ─────────────────────────────────────────────────────────────────────
class Perceptron(nn.Module):
    """Perceptron 1 lớp — không có hidden layer."""
    def __init__(self, n_in=2):
        super().__init__()
        self.fc = nn.Linear(n_in, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class MLP_XOR(nn.Module):
    """MLP 1 hidden layer — học được XOR."""
    def __init__(self, n_hidden=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(model, X, y, epochs=5000, lr=0.5):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.BCELoss()
    losses, accs = [], []

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        with torch.no_grad():
            pred = (out > 0.5).float()
            accs.append((pred == y).float().mean().item())

    return losses, accs


def get_accuracy(model, X, y):
    with torch.no_grad():
        pred = (model(X).squeeze() > 0.5).float()
        return (pred == y).float().mean().item() * 100


# ── Decision boundary ──────────────────────────────────────────────────────────
def plot_decision_boundary(ax, model, X, y, title, subtitle=''):
    xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 300),
                          np.linspace(-0.3, 1.3, 300))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        zz = model(grid).squeeze().numpy().reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=50, cmap='RdYlGn', alpha=0.7, vmin=0, vmax=1)
    ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=2, linestyles='--')

    colors = ['#e74c3c', '#27ae60']
    labels_text = [f'[{int(xi[0])},{int(xi[1])}]→{int(yi)}' for xi, yi in zip(X.numpy(), y.numpy())]
    for i, (xi, yi) in enumerate(zip(X.numpy(), y.numpy())):
        ax.scatter(xi[0], xi[1], c=colors[int(yi)], s=250, zorder=5,
                   edgecolors='black', linewidths=1.5)
        ax.annotate(labels_text[i], (xi[0], xi[1]),
                    textcoords='offset points', xytext=(8, 6), fontsize=9)

    ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel('Input A', fontsize=10); ax.set_ylabel('Input B', fontsize=10)
    ax.set_title(f'{title}\n{subtitle}', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    p0 = mpatches.Patch(color='#e74c3c', label='Output = 0')
    p1 = mpatches.Patch(color='#27ae60', label='Output = 1')
    ax.legend(handles=[p0, p1], fontsize=8, loc='lower right')


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & KẾT QUẢ
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 60)
print('DEMO: MLP HỌC CÁC CỔNG LOGIC')
print('=' * 60)

EPOCHS = 5000
results = {}

# ── NOT gate (perceptron 1 input) ──────────────────────────────────────────────
print('\n[1] NOT gate — Perceptron 1 input:')
m_not = Perceptron(n_in=1)
l_not, a_not = train_model(m_not, X1, Y_NOT, EPOCHS, lr=0.5)
acc_not = get_accuracy(m_not, X1, Y_NOT)
print(f'    Accuracy: {acc_not:.1f}%  |  Final loss: {l_not[-1]:.6f}')
with torch.no_grad():
    for xi, yi in zip(X1, Y_NOT):
        pred = m_not(xi).item()
        print(f'    NOT({int(xi.item())}) = {pred:.3f} → {int(pred>0.5)}  (đúng: {int(yi.item())})')
results['NOT'] = {'losses': l_not, 'accs': a_not, 'acc': acc_not,
                  'model': 'Perceptron', 'sep': 'Phân tách tuyến tính ✓'}

# ── AND, OR, XOR với Perceptron ────────────────────────────────────────────────
models_perceptron = {}
for gate_name, (y_vals, sep_note) in GATES_2IN.items():
    y_t = torch.tensor(y_vals, dtype=torch.float32)
    model = Perceptron(n_in=2)
    losses, accs = train_model(model, X2, y_t, EPOCHS, lr=0.5)
    acc = get_accuracy(model, X2, y_t)
    print(f'\n[Perceptron] {gate_name} gate — {sep_note}')
    print(f'    Accuracy: {acc:.1f}%  |  Final loss: {losses[-1]:.6f}')
    with torch.no_grad():
        preds = model(X2).squeeze()
        for xi, yi, pi in zip(X2, y_t, preds):
            tag = '✓' if (pi > 0.5) == bool(yi) else '✗'
            print(f'    {gate_name}({int(xi[0])},{int(xi[1])}) = {pi:.3f} → {int(pi>0.5)}  '
                  f'(đúng: {int(yi)}) {tag}')
    results[f'{gate_name}_perceptron'] = {'losses': losses, 'accs': accs,
                                           'acc': acc, 'model': 'Perceptron',
                                           'sep': sep_note, 'y': y_t}
    models_perceptron[gate_name] = model

# ── XOR với MLP (giải pháp) ───────────────────────────────────────────────────
print(f'\n[MLP 1 hidden layer] XOR gate — Giải pháp cho vấn đề trên:')
y_xor = torch.tensor(GATES_2IN['XOR'][0], dtype=torch.float32)
m_mlp = MLP_XOR(n_hidden=4)
l_mlp, a_mlp = train_model(m_mlp, X2, y_xor, EPOCHS, lr=0.5)
acc_mlp = get_accuracy(m_mlp, X2, y_xor)
print(f'    Accuracy: {acc_mlp:.1f}%  |  Final loss: {l_mlp[-1]:.6f}')
with torch.no_grad():
    preds = m_mlp(X2).squeeze()
    for xi, yi, pi in zip(X2, y_xor, preds):
        tag = '✓' if (pi > 0.5) == bool(yi) else '✗'
        print(f'    XOR({int(xi[0])},{int(xi[1])}) = {pi:.3f} → {int(pi>0.5)}  '
              f'(đúng: {int(yi)}) {tag}')
results['XOR_mlp'] = {'losses': l_mlp, 'accs': a_mlp, 'acc': acc_mlp,
                      'model': 'MLP (1 hidden)', 'sep': 'Giải quyết bằng hidden layer ✓'}

# ── Lưu model MLP-XOR để dùng tham khảo ──────────────────────────────────────
torch.save(m_mlp.state_dict(), f'{OUT_DIR}/mlp_xor.pth')
print(f'\n✓ Đã lưu MLP-XOR model: {OUT_DIR}/mlp_xor.pth')


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 1: Loss Curves
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('MLP Học Các Cổng Logic — Loss & Accuracy theo Epoch',
             fontsize=14, fontweight='bold')

plot_configs = [
    ('NOT_perceptron_proxy', 'NOT Gate\n(Perceptron 1 input)',
     results['NOT']['losses'], results['NOT']['accs'],
     '#3498db', results['NOT']['acc']),
    ('AND_perceptron', 'AND Gate\n(Perceptron)',
     results['AND_perceptron']['losses'], results['AND_perceptron']['accs'],
     '#2ecc71', results['AND_perceptron']['acc']),
    ('OR_perceptron', 'OR Gate\n(Perceptron)',
     results['OR_perceptron']['losses'], results['OR_perceptron']['accs'],
     '#f39c12', results['OR_perceptron']['acc']),
    ('XOR_perceptron', 'XOR Gate\n(Perceptron — THẤT BẠI)',
     results['XOR_perceptron']['losses'], results['XOR_perceptron']['accs'],
     '#e74c3c', results['XOR_perceptron']['acc']),
    ('XOR_mlp', 'XOR Gate\n(MLP 1 hidden layer — THÀNH CÔNG)',
     results['XOR_mlp']['losses'], results['XOR_mlp']['accs'],
     '#9b59b6', results['XOR_mlp']['acc']),
]

ep = range(1, EPOCHS + 1)
for ax_row in range(2):
    for ax_col, (key, title, losses, accs, color, acc) in enumerate(plot_configs[:3] if ax_row == 0 else plot_configs[3:]):
        ax = axes[ax_row][ax_col if ax_row == 0 else ax_col]
        if ax_row == 1 and ax_col == 2:
            ax.axis('off')
            continue
        ax2 = ax.twinx()
        ax.plot(ep, losses, color=color, lw=1.5, label='Loss')
        ax2.plot(ep, accs, color='gray', lw=1.5, linestyle='--', label='Accuracy')
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Loss (BCELoss)', fontsize=9, color=color)
        ax2.set_ylabel('Accuracy', fontsize=9, color='gray')
        ax2.set_ylim(0, 1.1)
        ax.set_title(f'{title}\nFinal Acc: {acc:.1f}%', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

# Panel cuối: bảng tổng kết
ax_summary = axes[1][2]
ax_summary.axis('off')
summary_data = [
    ['AND', 'Perceptron', f'{results["AND_perceptron"]["acc"]:.0f}%', '✓'],
    ['OR',  'Perceptron', f'{results["OR_perceptron"]["acc"]:.0f}%',  '✓'],
    ['NOT', 'Perceptron', f'{results["NOT"]["acc"]:.0f}%',            '✓'],
    ['XOR', 'Perceptron', f'{results["XOR_perceptron"]["acc"]:.0f}%', '✗'],
    ['XOR', 'MLP (hidden)', f'{results["XOR_mlp"]["acc"]:.0f}%',      '✓'],
]
tbl = ax_summary.table(
    cellText=summary_data,
    colLabels=['Gate', 'Model', 'Accuracy', 'Kết quả'],
    cellLoc='center', loc='center'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
    elif summary_data[r-1][3] == '✓':
        cell.set_facecolor('#d5f5e3')
    else:
        cell.set_facecolor('#fadbd8')
ax_summary.set_title('Tổng kết', fontsize=11, fontweight='bold')

plt.tight_layout()
path1 = f'{OUT_DIR}/loss_curves.png'
plt.savefig(path1, dpi=150, bbox_inches='tight')
plt.close()
print(f'✓ Loss curves: {path1}')


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 2: Decision Boundaries
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Decision Boundary — Ranh Giới Phân Loại',
             fontsize=13, fontweight='bold')

y_and = results['AND_perceptron']['y']
y_or  = results['OR_perceptron']['y']
y_xor_t = results['XOR_perceptron']['y']

plot_decision_boundary(axes[0], models_perceptron['AND'], X2, y_and,
    'AND Gate', f'Perceptron — Acc {results["AND_perceptron"]["acc"]:.0f}%')
plot_decision_boundary(axes[1], models_perceptron['OR'], X2, y_or,
    'OR Gate', f'Perceptron — Acc {results["OR_perceptron"]["acc"]:.0f}%')
plot_decision_boundary(axes[2], models_perceptron['XOR'], X2, y_xor_t,
    'XOR Gate ← THẤT BẠI', f'Perceptron — Acc {results["XOR_perceptron"]["acc"]:.0f}%\n'
    'Đường thẳng không thể phân tách XOR!')
plot_decision_boundary(axes[3], m_mlp, X2, y_xor_t,
    'XOR Gate ← THÀNH CÔNG', f'MLP 1 hidden layer — Acc {results["XOR_mlp"]["acc"]:.0f}%\n'
    'Hidden layer tạo ranh giới phi tuyến!')

plt.tight_layout()
path2 = f'{OUT_DIR}/decision_boundaries.png'
plt.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f'✓ Decision boundaries: {path2}')


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 3: So sánh XOR Perceptron vs MLP (slide chính)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('XOR: Tại sao Perceptron thất bại và MLP thành công?',
             fontsize=13, fontweight='bold')

# Panel 1: Truth table XOR visualization
ax = axes[0]
ax.set_title('Bài toán XOR\n(Không phân tách tuyến tính)', fontsize=11, fontweight='bold')
points = [(0,0,0,'#e74c3c'),(0,1,1,'#27ae60'),(1,0,1,'#27ae60'),(1,1,0,'#e74c3c')]
for x, y, label, c in points:
    ax.scatter(x, y, c=c, s=400, zorder=5, edgecolors='black', linewidths=2)
    ax.annotate(f'({x},{y})→{label}', (x, y), textcoords='offset points',
                xytext=(12, 8), fontsize=12, fontweight='bold')

ax.annotate('', xy=(0.7, 0.8), xytext=(0.1, 0.1),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax.text(0.25, 0.5, 'Không tồn tại\nđường thẳng nào\nphân tách được!',
        fontsize=10, color='#c0392b', ha='center',
        bbox=dict(boxstyle='round', facecolor='#fadbd8', alpha=0.8))
ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4)
ax.set_xlabel('Input A', fontsize=11); ax.set_ylabel('Input B', fontsize=11)
ax.grid(alpha=0.3)
p0 = mpatches.Patch(color='#e74c3c', label='Output = 0')
p1 = mpatches.Patch(color='#27ae60', label='Output = 1')
ax.legend(handles=[p0, p1], fontsize=10)

# Panel 2: Loss curve comparison
ax = axes[1]
ax.plot(range(1, EPOCHS+1), results['XOR_perceptron']['losses'],
        color='#e74c3c', lw=2, label=f'Perceptron — Acc {results["XOR_perceptron"]["acc"]:.0f}%')
ax.plot(range(1, EPOCHS+1), results['XOR_mlp']['losses'],
        color='#9b59b6', lw=2, label=f'MLP 1 hidden — Acc {results["XOR_mlp"]["acc"]:.0f}%')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss (BCELoss)', fontsize=11)
ax.set_title('Training Loss theo Epoch\nPerceptron vs MLP (XOR)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.text(EPOCHS*0.6, max(results['XOR_perceptron']['losses'])*0.85,
        'Perceptron:\nLoss không về 0\n→ Không học được XOR',
        color='#e74c3c', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#fadbd8', alpha=0.8))
ax.text(EPOCHS*0.4, results['XOR_mlp']['losses'][-1]*8,
        'MLP:\nLoss → 0\n→ Học được XOR!',
        color='#9b59b6', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#e8daef', alpha=0.8))

# Panel 3: Decision boundaries so sánh
ax = axes[2]
ax.set_title('Kiến trúc mạng\nPerceptron vs MLP', fontsize=11, fontweight='bold')
ax.axis('off')

arch_text = (
    "PERCEPTRON (thất bại với XOR):\n"
    "Input(2) ──→ Linear(2→1) ──→ Sigmoid\n"
    "Chỉ tạo được ranh giới THẲNG\n\n"
    "─" * 35 + "\n\n"
    "MLP (thành công với XOR):\n"
    "Input(2) ──→ Linear(2→4)\n"
    "           ──→ ReLU  ← hidden layer!\n"
    "           ──→ Linear(4→1) ──→ Sigmoid\n"
    "Tạo được ranh giới PHI TUYẾN\n\n"
    "─" * 35 + "\n\n"
    "Nguyên lý này mở rộng ra:\n"
    "BERT(384-dim) → MLP(256→128→6)\n"
    "→ Phân loại 6 ngành CV!"
)
ax.text(0.05, 0.95, arch_text, transform=ax.transAxes,
        fontsize=10.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#eaf2ff', alpha=0.9))

plt.tight_layout()
path3 = f'{OUT_DIR}/xor_analysis.png'
plt.savefig(path3, dpi=150, bbox_inches='tight')
plt.close()
print(f'✓ XOR analysis: {path3}')

# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('TỔNG KẾT KẾT QUẢ')
print('=' * 60)
rows = [
    ('NOT', 'Perceptron', results['NOT']['acc']),
    ('AND', 'Perceptron', results['AND_perceptron']['acc']),
    ('OR',  'Perceptron', results['OR_perceptron']['acc']),
    ('XOR', 'Perceptron', results['XOR_perceptron']['acc']),
    ('XOR', 'MLP 1 hidden', results['XOR_mlp']['acc']),
]
print(f'{"Gate":<6} {"Model":<20} {"Accuracy":>10} {"Kết quả"}')
print('-' * 50)
for gate, model, acc in rows:
    status = '✓ Học được' if acc >= 99 else '✗ Không học được'
    print(f'{gate:<6} {model:<20} {acc:>9.1f}%  {status}')

print(f'\n✓ Tất cả hình ảnh lưu tại: {OUT_DIR}/')
print('  - loss_curves.png         : Loss & Accuracy theo epoch')
print('  - decision_boundaries.png : Ranh giới phân loại 2D')
print('  - xor_analysis.png        : Phân tích XOR chi tiết (dùng cho PPT)')
print('\nKết luận quan trọng:')
print('  XOR cần hidden layer vì không phân tách tuyến tính.')
print('  Nguyên lý này áp dụng vào CV screening:')
print('  BERT(384) → MLP(256→128→6) học phân loại 6 ngành phức tạp.')


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: TRAIN GATE MLP MODELS — TÍCH HỢP VÀO ỨNG DỤNG THỰC TẾ
# Input: continuous BERT skill scores [0.0, 1.0] thay vì binary [0, 1]
# MLP thật sự học ranh giới phân loại từ dữ liệu — không phải if/else
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('PHẦN 2: TRAIN GATE MLP CHO ỨNG DỤNG THỰC TẾ')
print('Input: BERT skill scores liên tục [0.0, 1.0]')
print('=' * 60)

BERT_THRESHOLD = 0.28   # ngưỡng check_skill_bert — MLP học ranh giới này
N_SAMPLES      = 30000  # đủ để MLP học boundary mượt
N_HIDDEN_GATE  = 8      # hidden neurons cho gate MLP


class GateMLP(nn.Module):
    """MLP học cổng logic từ continuous BERT skill scores."""
    def __init__(self, n_in=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, N_HIDDEN_GATE),
            nn.ReLU(),
            nn.Linear(N_HIDDEN_GATE, 1),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def make_gate_data(gate_type: str, n=N_SAMPLES, thr=BERT_THRESHOLD):
    """
    Sinh dữ liệu training từ continuous BERT scores.
    Score ~ Uniform[0,1], label theo logic cổng với ngưỡng thr.
    """
    rng = np.random.default_rng(42)
    if gate_type == 'NOT':
        X = rng.uniform(0, 1, (n, 1)).astype(np.float32)
        # NOT: nếu kỹ năng KHÔNG có (score < thr) → pass (1); nếu có → không áp dụng NOT (0)
        y = (X[:, 0] < thr).astype(np.float32)
    else:
        X = rng.uniform(0, 1, (n, 2)).astype(np.float32)
        a = X[:, 0] >= thr
        b = X[:, 1] >= thr
        if gate_type == 'AND':
            y = (a & b).astype(np.float32)
        elif gate_type == 'OR':
            y = (a | b).astype(np.float32)
        else:  # XOR
            y = (a ^ b).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)


def train_gate(gate_name: str):
    n_in   = 1 if gate_name == 'NOT' else 2
    X, y   = make_gate_data(gate_name)
    model  = GateMLP(n_in=n_in)
    opt    = torch.optim.Adam(model.parameters(), lr=5e-3)
    crit   = nn.BCELoss()
    losses = []

    for epoch in range(3000):
        # Mini-batch training
        idx   = torch.randperm(len(X))[:512]
        xb, yb = X[idx], y[idx]
        opt.zero_grad()
        loss = crit(model(xb).squeeze(), yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    with torch.no_grad():
        pred = (model(X).squeeze() > 0.5).float()
        acc  = (pred == y).float().mean().item() * 100
    return model, acc, losses


gate_trained = {}
for gname in ['AND', 'OR', 'NOT', 'XOR']:
    model_g, acc_g, losses_g = train_gate(gname)
    gate_trained[gname] = (model_g, acc_g, losses_g)

    # Lưu model
    save_path = f'{OUT_DIR}/gate_{gname.lower()}_mlp.pth'
    torch.save({
        'state_dict': model_g.state_dict(),
        'n_in':       1 if gname == 'NOT' else 2,
        'threshold':  BERT_THRESHOLD,
        'gate':       gname,
    }, save_path)
    print(f'  {gname} gate MLP — Accuracy: {acc_g:.2f}%  →  {save_path}')

    # Kiểm tra trên truth table (test generalization)
    if gname != 'NOT':
        test_X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        # map binary → scores: 0→0.0, 1→0.9 (mô phỏng BERT score cao/thấp)
        test_X_scaled = test_X * 0.9
        with torch.no_grad():
            out = model_g(test_X_scaled).squeeze()
        expected = {'AND':[0,0,0,1],'OR':[0,1,1,1],'XOR':[0,1,1,0]}[gname]
        print(f'    Truth table (0→0.0, 1→0.9):')
        for xi, oi, ei in zip(test_X.numpy(), out.numpy(), expected):
            tag = '✓' if (oi > 0.5) == bool(ei) else '✗'
            print(f'      {gname}({int(xi[0])},{int(xi[1])}) '
                  f'= {oi:.3f} → {int(oi>0.5)}  (đúng: {ei}) {tag}')
    else:
        test_X = torch.tensor([[0.0],[0.9]])
        with torch.no_grad():
            out = model_g(test_X).squeeze()
        print(f'    NOT(0.0) = {out[0]:.3f} → {int(out[0]>0.5)}  (đúng: 1)')
        print(f'    NOT(0.9) = {out[1]:.3f} → {int(out[1]>0.5)}  (đúng: 0)')


# ── Vẽ loss curves của 4 gate models ─────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle('Training Loss — Gate MLP Models (Continuous BERT Scores)',
             fontsize=12, fontweight='bold')
colors_g = {'AND':'#2ecc71','OR':'#f39c12','NOT':'#3498db','XOR':'#9b59b6'}

for ax, gname in zip(axes, ['AND','OR','NOT','XOR']):
    _, acc_g, losses_g = gate_trained[gname]
    ax.plot(losses_g, color=colors_g[gname], lw=1.5)
    ax.set_title(f'{gname} Gate MLP\nAcc: {acc_g:.2f}%', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
    ax.grid(alpha=0.3)
    ax.text(0.6, 0.8, f'Loss → {losses_g[-1]:.4f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
path4 = f'{OUT_DIR}/gate_mlp_training.png'
plt.savefig(path4, dpi=150, bbox_inches='tight')
plt.close()
print(f'\n✓ Gate MLP training curves: {path4}')
print('\nPhần 2 hoàn tất — 4 gate models đã lưu, sẵn sàng tích hợp vào ứng dụng.')
