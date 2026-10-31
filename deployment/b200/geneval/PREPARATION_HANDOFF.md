# FMRG preparation update — 2026-11-01

**Chưa launch full.** Đây là phần bổ sung cho [handoff core/ensemble trước](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/4973c68ccc4e088f5634e53c0dd63e38f0d17e65/full-paper/20261029/FULL_PAPER_HANDOFF.md). Core runtime/source và các artifacts immutable cũ được giữ nguyên. Các thay đổi dưới đây thay thế hướng dẫn GenEval/VLM/splits đã cũ trong handoff29/10.

## Đã chuẩn bị

- **GenEval overlay** tại `/opt/fmrg-geneval`, phụ thuộc core `/opt/fmrg` SHA256 `06ef365e64d7dcccd7507ac3435f46bf25b724679b65c8e1a8d7a060958f43ff`. Overlay162,103,715bytes, SHA256 `16126b76c1540e464cdef993f0e7551432f7203d8de000f470244edeabd0b7f1`. Không ghi đè venv Torch/core.
- PyTorch2.7.1+cu128/Python3.11.14 kế thừa core; MMCV-full1.7.1 build native sm_100 với CUDA12.8.1, MMDetection2.28.2, open_clip2.20/timm.9.2 trong evaluator overlay. Source MMCV chỉ đổi compiler C++14→C++17, không đổi numerical kernels. Đây là adapted ABI environment, không phải bản môi trường Torch1.12/cu113 gốc.
- **Modal B200 technical PASS:** native deformable attention forward/backward đối chiếu CPUreference, CUDA NMS, cả detector R50/Swin-S và official GenEval color-classifier offline. Strict checkpoint key/shape audit: R50 610/610, Swin-S649/649; không thiếu/thừa/sai shape. Giữ OpenAI loader và36QuickGELU modules; chỉ thay cache path. TCP bị chặn trong canary. Đây không phải dataset score hoặc chứng nhận company node.
- **Skywork** đã cấp quyền và mirror17files nguồn + manifest, khoảng16.6GB, revision `cfc6496e451f4e636d7e78ce6249a4ee0efb89a1`. Blocker access đóng; paper Yes/No-logit adapter vẫn chưa qualify, không tráo bằng scalar valuehead.
- **Reconstructed split** đã được user cho phép: AFHQ-Dog original V1 mirror và FFHQ aligned1024→256, mỗi dataset50tuning+1000eval, tổng2100PNG. Sort SHA256(namespace/dataset/ID), first50 tuning,next1000 eval; không adaptive replacement. Freeze source revisions, original IDs, raw/decoded/preprocessed/PNG hashes và preprocessing. Đã verify FFHQ từng raw image theo MD5 metadata gốc NVIDIA và giữ attribution/license. Không phải author-exact; measurement tensors chưa sinh.

## Bootstrap qua HF/proxy

Operator/agent lấy path, proxy, allocation/GPU UUID/deadline từ runbook/receipt đang hiệu lực. Dùng layout `fmrg_all` đã duyệt; output/toolkit mới, không overwrite attempts hoặc sửa jobs đang chạy. Không tải/cài qua GitHub/PyPI trên node; không gọi Modal. Không log secrets hoặc proxy có auth.

- [Publication manifest](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/08b0ac3905affcb2a1439c05629d451a956885eb/full-paper/20261101/publication.json) — SHA256 `e61ee0b366b751a57ebddd5b8674544309f5a893381325c464a6c59179a1c610`.
- [Toolkit](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/5a5cc2c18fdfccc5819dd92c0db49bdb543c81a7/full-paper/20261101/geneval-toolkit-20261101.tar.gz) — SHA256 `2e436b0fd5aaf1b0d077faad271711e4912950a11faf47eca040e743947c2daf`.
- [Contract update](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/5a5cc2c18fdfccc5819dd92c0db49bdb543c81a7/full-paper/20261101/CONTRACT_UPDATE.json). Handoff này và publication đều giữ `full_launch_allowed=false`.

Trong terminal công ty, đặt `FMRG_PROXY`, `FMRG_PREP_DOWNLOADS` là thư mục tuyệt đối có sẵn; `FMRG_PREP_TOOLKIT` là thư mục mới, parent tồn tại. Dùng một archive/helper mới trong download directory nếu attempt cũ còn đó. Không tự mở allocation mới.

```bash
set -euo pipefail
: "${FMRG_PROXY:?approved HF proxy}"
: "${FMRG_PREP_DOWNLOADS:?existing approved absolute download directory}"
: "${FMRG_PREP_TOOLKIT:?new approved toolkit directory}"
: "${FMRG_EXTRA_ASSETS:?approved extra asset root}"
test -d "$FMRG_PREP_DOWNLOADS"
test ! -e "$FMRG_PREP_TOOLKIT"
toolkit_archive="$FMRG_PREP_DOWNLOADS/geneval-toolkit-20261101.tar.gz"
test ! -e "$toolkit_archive"
curl -fL --retry 2 --proxy "$FMRG_PROXY" --noproxy '' 'https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/5a5cc2c18fdfccc5819dd92c0db49bdb543c81a7/full-paper/20261101/geneval-toolkit-20261101.tar.gz' -o "$toolkit_archive"
printf '%s  %s\n' '2e436b0fd5aaf1b0d077faad271711e4912950a11faf47eca040e743947c2daf' "$toolkit_archive" | sha256sum -c -
mkdir "$FMRG_PREP_TOOLKIT"
tar -xzf "$toolkit_archive" -C "$FMRG_PREP_TOOLKIT" --no-same-owner
python3 "$FMRG_PREP_TOOLKIT/download_publication.py" \
  --manifest-url 'https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/08b0ac3905affcb2a1439c05629d451a956885eb/full-paper/20261101/publication.json' --sha256 'e61ee0b366b751a57ebddd5b8674544309f5a893381325c464a6c59179a1c610' \
  --dest "$FMRG_EXTRA_ASSETS" --group runtime --group reconstructed
```

Runtime group có overlay, wheel, build/preflight/canary receipts. Reconstructed group có223MBarchive +16MBmanifest/selection/licenses. Muốn stage Skywork riêng, gọi lại helper với `--group vlm`; dự trù thêm16.6GB. Public URLs không cần token; hash verification không cấp quyền dùng GPU hoặc chứng nhận reward adapter.

## Restore và canary trên company B200

Tái dùng core runtime đã verify và receipts canary hiện hành. Source frozen vẫn `be485ba40f1d1c163b95ad61112aca6baa13ed22`; giữ artifacts/data tách khỏi runtime. Restore helper chỉ kiểm core version/ABI và archive, còn core hash/gate receipt phải được agent/operator xác nhận từ evidence cũ.

```bash
/opt/fmrg/venv/bin/python "$FMRG_PREP_TOOLKIT/restore_overlay.py" \
  --archive "$FMRG_EXTRA_ASSETS/runtime/geneval-overlay-v2.tar.gz" \
  --sha256 '16126b76c1540e464cdef993f0e7551432f7203d8de000f470244edeabd0b7f1' \
  --bytes 162103715
```

Helper từ chối `/opt/fmrg-geneval` tồn tại, unsafe tar paths/links hoặc coreABI sai. Không sudo, install hoặc sửa driver tự động. Nếu prefix đã có đúng overlay/receipt, verify/reuse và tạo run mới; không restore đè. Python3.11+ từ runtime core dùng được để chạy helper nếu Python hệ thống thiếu API tar filter.

`FMRG_OLD_EXTRA_ASSETS` phải trỏ extra-assets29/10 đã verify, chứa `geneval/{ViT-L-14.pt,mask2former_r50_lsj_8x2_50e_coco.pth,mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth}`. Đây là tên rút gọn thực tế trong bundle; hướng dẫn basename có timestamp ở handoff cũ là sai. Giữ weights/hash cũ, không rename theo suy đoán.

Chỉ chạy lệnh GPU dưới đây trong allocation đúng một B200 đang được cấp, theo queue hiện hành; giữ CUDA_VISIBLE_DEVICES. Đặt `FMRG_GENEVAL_RUN` là output mới, parent đã duyệt. Thu thêm allocation/job/GPU UUID/driver receipt từ scheduler; Modal canary không cung cấp UUID company.

```bash
: "${FMRG_OLD_EXTRA_ASSETS:?verified extra-assets29/10 root}"
: "${FMRG_GENEVAL_RUN:?new approved canary output directory}"
test ! -e "$FMRG_GENEVAL_RUN"
mkdir "$FMRG_GENEVAL_RUN"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
timeout 180 /opt/fmrg-geneval/venv/bin/python "$FMRG_PREP_TOOLKIT/canary.py" \
  --assets "$FMRG_OLD_EXTRA_ASSETS" --out "$FMRG_GENEVAL_RUN/canary.json" \
  > "$FMRG_GENEVAL_RUN/canary.log" 2>&1
```

Canary dùng một demo fixture pinned từ MMDetection và synthetic color patch; không sinh ảnh FLUX, không chạy553prompts×4seeds. Receipt có native diffs, checkpoint/config/source identities, finite detections, QuickGELU, elapsed/VRAM. Import-only, timeout hoặc thiếu JSON không được ghi PASS. R50 là profile chính theo paper; Swin-S là official-default control, giữ score tách biệt. Chưa chứng minh numerical equivalence với evaluator Torch1.12/cu113 gốc.

## Data/evidence và blockers còn lại

Reconstructed archive chỉ có ảnh/metadata, không measurement tensors. Freeze operator/kernel/noise-seed/tensor hashes trước tuning. Không trộn reconstructed với author-exact hoặc đổi split theo metric. `strict_evidence.py` mặc định yêu cầu đủ measurement hashes; image-only validation phải khai báo rõ `require_measurements=False`. GenEval aggregation reject missing/duplicate/failed/metadata-mismatched cells và yêu cầu đủ6categories, poolseeds trước mean category scores; không zero-fill.

Author-exact splits/style16refs+48pairedprompts vẫn chưa có. Paper-formula adapter, VLM differentiable Yes/No adapter/renoiseconstantc, complete tuning objective/grid/baselines, instrumentation và full allocation/deadline chưa chốt. `tuning-plan-draft.json` chỉ là đề xuất độc lập có nhãn reconstructed, chưa run/chưa lấy lại author search budget. Ensemble B200 company canary đang do agent khác kiểm, chưa nhận raw receipt ở task này.

Trả report bỏ secrets: downloaded groups/hashes, runtime/core/source identities, allocation/job/GPU, canary JSON/log/exitcode/time/VRAM, trạng thái từng blocker có evidence path, kế hoạch bước kế tiếp. Không chạy full matrix, không rerun shortlist, không gọi source generators hoặc tự nâng launch flag. Full execution cần chỉ thị riêng.
