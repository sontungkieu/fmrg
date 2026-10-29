# Full-paper preparation handoff — 2026-10-29

**Chuẩn bị và audit đã xong trong phạm vi có thể xác minh; chưa được launch full.** `CONTRACT.json` vẫn có `full_launch_allowed=false`. Verdict company screening đã chốt: hoàn tất replay screening, không bitwise, chưa xác nhận tương đương phân phối; số liệu company được lưu theo báo cáo người dùng, chưa raw-audit độc lập.

## Identities và tải về

- Core runtime/setup cũ giữ nguyên: [handoff core](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/6352766e35c892ec51d70dcf79e5c4968a15de7e/HANDOFF.md). Runtime `/opt/fmrg`, source `be485ba40f1d1c163b95ad61112aca6baa13ed22`, Torch2.7.1+cu128/SDPA. Tái dùng receipt đã đạt ở node; không restore đè hoặc rerun shortlist chỉ vì có handoff mới.
- [Toolkit + audit + contract + evidence](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/77f4ae1dee744fbfa906eaf3f9cd00b66a344594/full-paper/20261029/full-paper-toolkit.tar.gz), SHA256 `3820e3fc6948381200beb640df71b3e06cbee8d404c0110efed55b9b34f21525`, bytes `26840`.
- [CONTRACT.json](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/77f4ae1dee744fbfa906eaf3f9cd00b66a344594/full-paper/20261029/toolkit/CONTRACT.json), SHA256 `1b02425b2c9d6c447bc7350f2d168c35af6b2e06bf6f904d43ca745eb098b3cb`.
- [Audit đọc được](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/77f4ae1dee744fbfa906eaf3f9cd00b66a344594/full-paper/20261029/toolkit/PAPER_SOURCE_AUDIT.md); [asset manifest](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/10eb27b54ba7db3d44637efe59f8101f2edd92c3/full-paper/20261029/asset-manifest.json), SHA256 `a3963d58f982ec28f1a97cb512a94067838d27be235bc23a381eb3649d77dcc4`.
- Assets mới30files,10,153,895,280bytes (~9.46GiB), tải trực tiếp shared storage; dự trù thêm nguồn giải nén/cache/receipt, kiểm dung lượng thật. Không qua laptop. Core FLUX/LoRA/CLIP-H không tải lại nếu receipts/hash còn đúng.
- [Paper PDF qua HF](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/10eb27b54ba7db3d44637efe59f8101f2edd92c3/full-paper/20261029/assets/paper/2604.27147v3.pdf), SHA256 `640ff820d4b7dea270ebf8687ff29dbc476d9c08d63af0294f6b564eb8de2281`, CC BY4.0, Huang/Lin/Shah/Nair/Boffi, arXiv2604.27147v3. Giữ terms/model cards của từng component; public mirror không thay license upstream. Không gán một license chung cho mọi model/dataset.

## Công việc được giao cho agent công ty

Đọc AGENTS/runbook đang hiệu lực. Dùng đúng project `fmrg_all` đã được duyệt; lấy storage, queue, allocation, GPU UUID, deadline và proxy từ môi trường/receipt hiện tại. Giữ `CUDA_VISIBLE_DEVICES` của scheduler, không can thiệp job khác hoặc mở allocation mới. Company chỉ tải HF qua proxy được phép; không fallback GitHub/PyPI/nguồn khác. Không log token/proxy có auth, không bật `set -x`.

Chuẩn bị CPU trước; GPU canary chỉ trong allocation đang được cấp. Không gọi launcher Modal trong gói này. Không chạy `solve_inverse_problem.py`, `generate_geneval.py`, sweep hoặc full matrix ở lượt chuẩn bị này. Không tự sửa scientific source frozen để ép kết quả; nếu sau này triển khai paper-formula adapter thì ghi overlay/commit/manifest riêng và kiểm chứng trước.

## Setup toolkit/assets — CPU only

Operator đặt `FMRG_PROXY` và các đích tuyệt đối đã được duyệt. `FMRG_TOOLKIT` là thư mục mới, `FMRG_TOOLKIT_PARENT` tồn tại; `FMRG_ASSET_ROOT` là vùng assets mới hoặc chỉ chứa file cùng hash đã tải bằng helper. `FMRG_CORE_ASSETS` trỏ thư mục core có `pilot-assets.json` (lấy từ `publication-used.json` cũ), không đoán từ tên folder.

```bash
set -euo pipefail
: "${FMRG_PROXY:?approved HF proxy}"
: "${FMRG_TOOLKIT_PARENT:?existing absolute parent}"
: "${FMRG_TOOLKIT:?new absolute toolkit directory}"
: "${FMRG_ASSET_ROOT:?approved absolute extra-assets directory}"
test -d "$FMRG_TOOLKIT_PARENT"
test ! -e "$FMRG_TOOLKIT"
archive="$FMRG_TOOLKIT_PARENT/full-paper-toolkit-20261029.tar.gz"
test ! -e "$archive"
curl -fL --retry 2 --proxy "$FMRG_PROXY" --noproxy '' \
  'https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/77f4ae1dee744fbfa906eaf3f9cd00b66a344594/full-paper/20261029/full-paper-toolkit.tar.gz' -o "$archive"
printf '%s  %s\n' '3820e3fc6948381200beb640df71b3e06cbee8d404c0110efed55b9b34f21525' "$archive" | sha256sum -c -
mkdir "$FMRG_TOOLKIT"
tar --extract --gzip --file "$archive" --directory "$FMRG_TOOLKIT" --no-same-owner
cd "$FMRG_TOOLKIT"
sha256sum CONTRACT.json
# Phải khớp 1b02425b2c9d6c447bc7350f2d168c35af6b2e06bf6f904d43ca745eb098b3cb trước khi dùng helper.
python3 download_assets.py \
  --manifest-url 'https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/10eb27b54ba7db3d44637efe59f8101f2edd92c3/full-paper/20261029/asset-manifest.json' \
  --sha256 'a3963d58f982ec28f1a97cb512a94067838d27be235bc23a381eb3649d77dcc4' \
  --dest "$FMRG_ASSET_ROOT" \
  --component hps --component imagereward --component pickscore --component bert \
  --component metrics --component geneval --component sources --component style --component paper
```

Helper dùng standard library, verify immutable manifest/fullSHA/bytes, giữ file tồn tại sai hash để điều tra. Không chọn `--component vlm`: manifest ghi16failedfiles, helper sẽ từ chối trước khi tải. `verify_assets.py` rehash components trước sử dụng. `download-receipt.json` chỉ chứng nhận downloads, không phải neural canary.

## Canary evaluators và receipts

Các lệnh sau dùng runtime sẵn có, không pip/conda/rebuild. `FMRG_EVAL_RUN` phải là output mới trong vùng đã duyệt; hai kind ghi file mới và FAIL receipt nếu lỗi. `PYTHONPATH` chỉ overlay toolkit để imports helper; scientific module vẫn từ wrapper `/opt/fmrg/source`. Sampling/evaluator chạy offline. CPU metric canary đã pass trên Modal; chạy lại trên node để xác minh đường dẫn/runtime thực tế khi cần.

```bash
: "${FMRG_EVAL_RUN:?new approved receipt directory}"
: "${FMRG_CORE_ASSETS:?existing verified core asset directory}"
test ! -e "$FMRG_EVAL_RUN"
mkdir "$FMRG_EVAL_RUN"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export FMRG_SOURCE_ROOT=/opt/fmrg/source
export PYTHONPATH="$FMRG_TOOLKIT${PYTHONPATH:+:$PYTHONPATH}"
/opt/fmrg/python-fmrg.sh "$FMRG_TOOLKIT/canary_evaluators.py" metrics \
  --assets "$FMRG_ASSET_ROOT" --device cpu --out "$FMRG_EVAL_RUN/metrics.json" \
  > "$FMRG_EVAL_RUN/metrics.log" 2>&1
# Chỉ chạy trong queue/allocation đúng một B200 đã được cấp, timeout hữu hạn.
timeout 900 /opt/fmrg/python-fmrg.sh "$FMRG_TOOLKIT/canary_evaluators.py" reward \
  --assets "$FMRG_ASSET_ROOT" --core-assets "$FMRG_CORE_ASSETS" --device cuda \
  --out "$FMRG_EVAL_RUN/reward.json" > "$FMRG_EVAL_RUN/reward.log" 2>&1
```

Reward helper: path-only loader, tái dùng forward/preprocessing của released FMRG; HPS full checkpoint strictload, ImageReward local med_config/BERTtokenizer, CLIP/PickScore local_files_only. Không sinh ảnh FLUX. B200 four-model input-gradient canary **chưa chạy** ở gói này; nếu fail giữ traceback/receipt, không coi weights staged là pass. ImageReward upstream loader dùng strict=False; helper kiểm missing/unexpected keys trước khi nhận model. CPU metric kiểm LPIPS identity/gradient và2FID features; không tính FIDscore hoặc efficacy.

GenEval: tải đủ R50+Swin-S+OpenAI CLIP-L14 và source archive; generation runtime **không có mmcv/mmdet/open_clip ABI phù hợp**. Chỉ chuẩn bị source:

```bash
: "${FMRG_GENEVAL_SOURCE:?new absolute source directory}"
/opt/fmrg/python-fmrg.sh "$FMRG_TOOLKIT/prepare_geneval.py" \
  --assets "$FMRG_ASSET_ROOT" --dest "$FMRG_GENEVAL_SOURCE"
# Trong evaluator env đã có và được duyệt, nếu có:
# /absolute/approved/eval-python prepare_geneval.py --check-env
```

Imports pass còn cần native deformable-attention/detector inference canary trên hardware đích. Không tự install Torch1.12/cu113 vào B200 cu128 environment. Giữ isolated evaluator env/receipt nếu được cấp; nếu chưa có ghi BLOCKED. Paper-R50 config `configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py` và model basename `mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b` phải được chọn cùng nhau. Official default dùng Swin-S. OpenCLIP GenEval pretrained phải đổi **đường dẫn** sang CLIP-L14 đã verify, giữ nguyên transform/architecture; chưa có qualified wrapper nên không chạy evaluator rồi để nó tự tải internet.

## Contract còn chặn full

1. Author-exact AFHQ-Dog/FFHQ50tuning+1000eval IDs/hashes chưa có; không lấy1050ảnh tùy ý rồi gọi author split. Freeze raw/preprocessed/operator/noise/measurement hashes và disjoint tuning/eval. Không có full dataset mới trong bundle.
2. Hai track riêng: `released_code_v1` đã pin source; `paper_formula_v1` mới là audited specification. Loss, E weight, early-stop và Adam-J khác nhau; adapter paper chưa triển khai/qualify. Thực tế sigma grids và CPU call traces nằm trong evidence; không dùng NFEtag làm counter.
3. Ensemble B200 backward, GenEval native runtime/detector choice chưa chốt. LPIPS source[0,1]/224 khác standardprofile[-1,1]/native; ghi profile riêng. FID/KID count/hash/seed phải đủ, không dùng silent filename intersection.
4. Skywork weights/config/tokenizer gated; cần cấp quyền HF hợp lệ. Paper dùng Yes/No logits, không scalar valuehead. Adapter/renoiseconstantc chưa có. Style16refs+48pairedprompts/configs chưa có.
5. Complete tuning grid/objective/seeds, baseline runtime, all ablations, actual cost instrumentation và allocation/deadline chưa đủ. Các table settings đã được xác minh trong audit, không còn cần đoán paper; các artifact tác giả không công bố vẫn thiếu.

Trả một receipt bỏ secrets: HF revisions/SHA, contract track/hash, runtime/source identity, allocation/job/GPU, selected components/download hashes, canary JSON/log/exitcode, source/env imports và blockers. Không nâng `full_launch_allowed` hoặc launch full khi đóng các tác vụ chuẩn bị. Muốn full cần manifest mới đã review và chỉ thị launch riêng.
