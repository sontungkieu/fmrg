# Handoff: setup FMRG từ Hugging Face public

Cập nhật 28/10/2026. Đích: [codemaivanngu/fmrg-b200-portable](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable). Runtime, assets, source/helper và raw pilot outputs đã publish **public**; không cần HF token cho các URL này. Artifact commit: `73353f15eab0024925c942d197a7ed39a71aca0d`. Link dưới đây pin theo commit, không phụ thuộc `main`.

## Tải và setup

Trước chạy: operator xác nhận route/queue, đúng allocation 1 NVIDIA B200, Linux x86_64, driver hỗ trợ CUDA12.8, quyền ghi `/opt`, các thư mục được phép dùng và proxy HF hiện hành. Không lấy allocation/hostname lịch sử dự án khác. Giữ `CUDA_VISIBLE_DEVICES` của scheduler. Lệnh phải nằm trong queue job/container được phép; không chạy gates trên login node.

Runtime cố định tại `/opt/fmrg`; helper từ chối prefix/data/output đã tồn tại. Host cần curl, tar, sha256sum, Python3 và hỗ trợ libGL/glib (`libgl1`, `libglib2.0-0` trên Ubuntu). Dự trù khoảng 100 GiB trống cho archives và extraction; kiểm dung lượng thật trước chạy. Không cài Torch/CUDA/model trên laptop, không chạy apt hay đổi driver qua helper.

Đặt các biến bằng giá trị đã được operator duyệt: `FMRG_PROXY`, `FMRG_DOWNLOAD_DIR` (thư mục có sẵn), `FMRG_NEW_DATA_DIR` và `FMRG_NEW_RUN_DIR` (hai đích mới, parent đã tồn tại). Không ghi token/proxy có auth vào log, không bật `set -x`.

```bash
: "${FMRG_PROXY:?approved HF proxy}"
: "${FMRG_DOWNLOAD_DIR:?existing approved absolute directory}"
: "${FMRG_NEW_DATA_DIR:?new approved absolute directory}"
: "${FMRG_NEW_RUN_DIR:?new approved absolute directory}"
curl -fL --retry 2 --proxy "$FMRG_PROXY" --noproxy '' \
  'https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/setup_public.sh' \
  -o "$FMRG_DOWNLOAD_DIR/setup_public.sh"
printf '%s  %s\n' '85c167e873accd0ff0edccf9f49446bffeff120ecf8e92e0fbee27b41534dcba' "$FMRG_DOWNLOAD_DIR/setup_public.sh" | sha256sum -c -
bash "$FMRG_DOWNLOAD_DIR/setup_public.sh" \
  'https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/032f4d0b6191729a22a3b715f12a5ba59e9c66ad/publication.json' \
  '0418abaa347a14216168ceeb5f45fc52919ac8daebc657ad3f7c55755518f3bc' \
  "$FMRG_DOWNLOAD_DIR" "$FMRG_NEW_DATA_DIR" "$FMRG_NEW_RUN_DIR"
```

Helper kiểm manifest SHA, tải qua proxy và verify size/SHA từng bundle, restore runtime đúng prefix, verify từng asset, rồi chạy imports/kernel/tiny/full. Không tự launch scientific pilot/full benchmark. Chỉ ghi `FMRG_COMPANY_TECHNICAL_PASS` khi cả bốn gate thoát thành công. Thu thêm allocation/job ID để gắn receipts với GPU được cấp.

Manifest: [publication.json](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/032f4d0b6191729a22a3b715f12a5ba59e9c66ad/publication.json); SHA256 `0418abaa347a14216168ceeb5f45fc52919ac8daebc657ad3f7c55755518f3bc`. Manifest commit `032f4d0b6191729a22a3b715f12a5ba59e9c66ad` trỏ các artifact tại commit phía trên.

## Link tải và identity

| File | Bytes | SHA256 |
|---|---:|---|
| [runtime/runtime.tar.gz](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/runtime/runtime.tar.gz) | 4,275,503,585 | `06ef365e64d7dcccd7507ac3435f46bf25b724679b65c8e1a8d7a060958f43ff` |
| [assets/assets.tar](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/assets/assets.tar) | 38,783,272,960 | `4632c8d2471f5152eccd5e3c32c570389d1c7973dc9fbfae06931fd92260b1ab` |
| [manifests/archive-audit.json](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/manifests/archive-audit.json) | 1,618 | `52db5d07c95296b63c868051c6d981a6169b9817df940a4afbf95920e22701e4` |
| [manifests/assets-manifest.json](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/manifests/assets-manifest.json) | 8,082 | `f15dd4d3c19df56f7ff5fb24c76d906a464cfe790018f85462e9abab15c6427c` |
| [results/pilot-results.tar.gz](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/results/pilot-results.tar.gz) | 25,483,498 | `5b26bbeb897f59f8c1c7d16b59dcc7c63eaaaf1447a58a7a0cc7b1d481f6cbc7` |
| [source/setup-source.tar.gz](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/source/setup-source.tar.gz) | 20,707 | `f8a08684d2dca071340706c746dd7e1867931f8866bcc5cf6cdbab11a95682da` |
| [manifests/source-receipt.json](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/manifests/source-receipt.json) | 153 | `ad42786d14c107ccfb04d97b0e497b82beec71253e9b37b0075cad7787b03c04` |
| [setup_public.sh](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/setup_public.sh) | 5,946 | `85c167e873accd0ff0edccf9f49446bffeff120ecf8e92e0fbee27b41534dcba` |
| [README.md](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/README.md) | 3,934 | `38b21d47d08fc38802d1c2562b164de8942077e1c2c209cc9eed399eb539b57c` |
| [results/PILOT_RESULTS.md](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/results/PILOT_RESULTS.md) | 3,451 | `f754965a94044e898a745c59baa57023ff4024f0f5d8aa784fe36b07ac8fa5cf` |
| [manifests/pins.json](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/manifests/pins.json) | 1,496 | `ef3daa666b497595ee8dd61211a876a246c86f51e340504fc77800772241146c` |
| [licenses/LICENSE.md](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/73353f15eab0024925c942d197a7ed39a71aca0d/licenses/LICENSE.md) | 18,621 | `6f7b3d94ddd94380a4dc2d1211ed57883d320dea96f4393b34d4a454041029e1` |

Runtime public được repack từ archive đã clean-restore trên Modal B200, parent SHA `56832c3a1ef71c6911a08c16d58e4993b92dc451abeb2adbcc87430d337b369b`. Bỏ cloud launchers và đường dẫn vận hành local trong tools; giữ nguyên bytes interpreter/dependencies/upstream source. Vì vậy public archive có SHA riêng. Không tuyên bố đã GPU-requalify archive public hoặc đã chạy company node. Agent cần các gate trên allocation thực tế.

Assets giữ nguyên archive/bytes gốc: FLUX diffusers, LoRA256/512, CLIP ViT-H, 3 input PNG và 6 frozen SR measurements. Nhãn PRIVATE trong manifest nội bộ là trạng thái export cũ; user đã cho phép public ngày 28/10. License FLUX gốc được ship trong assets và `licenses/`; public không thay điều khoản upstream. Không có full AFHQ/FFHQ datasets, HPSv2/ImageReward/PickScore/VLM7B hoặc full paper evaluators.

## Replay và tiến tới reproduce

Source upstream `jrrhuang/fmrg@be485ba40f1d1c163b95ad61112aca6baa13ed22`; Python3.11.14, Torch2.7.1+cu128, CUDA12.8, SDPA. Sau setup đọc `publication-used.json` trong run dir để lấy `pilot_root`, `data_root` và các identities đã dùng. Wrapper `/opt/fmrg/python-fmrg.sh` đặt source/libs đúng prefix. Sampling offline với `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `FMRG_SOURCE_ROOT=/opt/fmrg/source`, `FMRG_DATA_ROOT=<data_root>`.

Chỉ sau technical pass và xác nhận allocation/thời lượng cho replay, submit qua queue hiện hành, output mới cho mỗi attempt:

```bash
# PILOT_ROOT và FMRG_DATA_ROOT lấy từ publication-used.json; không tự chọn GPU.
export FMRG_SOURCE_ROOT=/opt/fmrg/source HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/opt/fmrg/python-fmrg.sh "$PILOT_ROOT/experiments/pilot_ip.py" I "$APPROVED_REPLAY_DIR/I" 1000
/opt/fmrg/python-fmrg.sh "$PILOT_ROOT/experiments/pilot_ip.py" J "$APPROVED_REPLAY_DIR/J" 1000
/opt/fmrg/python-fmrg.sh "$PILOT_ROOT/experiments/pilot_reward.py" "$APPROVED_REPLAY_DIR/R" 1000
```

Các runner này chạy đủ pilot, không có cờ one-cell. I/J/R cần đủ 72/96/24 unique cells, summary COMPLETE, raw metrics và outputs; so sáu measurement hashes với archive gốc. Pilot G: 288 configs/8 oracle checks. Kết quả Modal là screening, chưa reproduce full paper. R chỉ CLIP; K3 có extra selection cost. R counter `autograd_calls` chỉ hook `autograd.grad`, không đếm `Tensor.backward`; cần bổ sung trước full benchmark. J chưa bảo đảm bitwise equality (smoke clean-restore cũ max PNG channel difference7/255).

Trước full paper: pin tuning/eval splits cho AFHQ-Dog/FFHQ, resolve pixel-squared so với norm/latent loss và timestep/early-stop contract, stage full reward/LPIPS/FID/GenEval evaluator qua HF, ghi đủ forwards/backwards/reward/selection cost. Không dùng ba ảnh/two-prompt screening thay full benchmark hoặc suy headline3NFE cho mọi setting.

Agent trả receipt bỏ secrets gồm: HF commit/hashes; host OS/driver/GPU; allocation/queue/job ID; resolved paths/config; exit code/JSON/log từng gate; counts/metrics/failures nếu replay; blocker còn lại. Import/finite gradients là technical pass; không gọi setup xong là reproduce các bảng paper.
