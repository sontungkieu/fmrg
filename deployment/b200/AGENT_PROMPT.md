Bạn là agent trên node công ty, tiếp tục giai đoạn chuẩn bị full-paper FMRG.
Screening đã chốt; nhiệm vụ mới là đọc audit/contract, setup assets và canary
evaluators còn thiếu. CHƯA LAUNCH FULL, không rerun shortlist theo mặc định.

Nguồn chuẩn:
- Handoff HF immutable: https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/4973c68ccc4e088f5634e53c0dd63e38f0d17e65/full-paper/20261029/FULL_PAPER_HANDOFF.md
- SHA256 handoff: 6d989c53b299ff9768ca93f5093ee72e222e556eeb84e4f362bbd66fdbbe02b5
- Contract: https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/77f4ae1dee744fbfa906eaf3f9cd00b66a344594/full-paper/20261029/toolkit/CONTRACT.json
- SHA256 contract: 1b02425b2c9d6c447bc7350f2d168c35af6b2e06bf6f904d43ca745eb098b3cb
- GitHub provenance: https://github.com/sontungkieu/fmrg ; deployment/b200/full_paper/.
- Scientific source frozen: be485ba40f1d1c163b95ad61112aca6baa13ed22.

1. Đọc AGENTS/runbook đang có và handoff mới qua proxy HF được phép, verify SHA.
   Lấy allocation/queue/deadline/GPU UUID/storage/proxy từ receipts hiện hành.
   Dùng manager đang vận hành, giữ CUDA_VISIBLE_DEVICES, không sửa job khác hoặc
   mở allocation mới. Project layout fmrg_all đã duyệt; không đoán host/path.
   Không log secrets/proxy có auth, không bật set -x.

2. Giữ runtime /opt/fmrg và source frozen. Tái dùng core FLUX/LoRAs/CLIP-H cùng
   receipt canary đã đạt; kiểm identity thay vì restore đè. Source/config/helper
   mới là toolkit riêng. Đọc các lệnh CPU setup trong handoff; tải toolkit và
   ~10.15GB assets mới trực tiếp shared storage qua HF proxy, verify fullSHA.
   Không tải qua laptop, GitHub Raw/PyPI hoặc pip/conda/rebuild trên node.

3. Dùng helper download_assets.py đúng CLI trong handoff. HPSv2.1, ImageReward,
   PickScore, BERTtokenizer, metric/style/GenEval weights và pinned source đã
   public. VLM chưa có weights:16file gated, README thôi; không bypass gate hoặc
   tự chấp nhận điều khoản. Exact AFHQ/FFHQ splits và style pairs chưa có,
   không dựng split tùy ý rồi gọi author-exact.

4. CPU metric canary dùng existing /opt/fmrg/python-fmrg.sh; pure-Python metric
   overlay nằm ngoài frozen venv. Chạy four-model reward input-gradient canary
   chỉ trong allocation đúng1B200 đã được cấp, timeout hữu hạn, output mới.
   Thu JSON/log/exitcode/VRAM/GPU/job identity. Không sinh ảnh FLUX. FAIL thì giữ
   traceback đầu và receipt, không retry mù hoặc chuyển technical gate thành PASS.

5. GenEval đã stage cả R50 lẫn Swin-S/CLIP-L14/source. Prepare source và kiểm
   native env có sẵn nếu được cấp. Generation runtime chưa có mmcv/mmdet ABI;
   không tự cài. Paper dùng R50, official evaluator default Swin-S; phải ghi
   rõ config+checkpoint choice, chưa có native kernel/detector canary thì ghi
   BLOCKED. Không để evaluator tự tải từ internet.

6. Đọc PAPER_SOURCE_AUDIT.md và CONTRACT.json. Hai track released_code_v1 và
   paper_formula_v1 khác loss/Eweight/early-stop/Adam-J; paper adapter chưa được
   triển khai/qualify. Giữ actual time grid và measured compute counters, không
   dùng NFEtag. Paper TableE.2/E.17 đã audit; exactsplits/tuning/seeds/style/VLM
   adapter/baseline env/allocation vẫn cần artifact/decision. Không sửa sampler
   frozen hoặc đổi metric preprocessing để làm đẹp kết quả.

7. Trả report/receipt bỏ secrets với HF revisions/hash, contract track/hash,
   runtime/source, allocation/job/GPU, component hashes, canary JSON/log/exitcode,
   evaluator env status, expected/actual outputs nếu có và blockers. Screening
   verdict: completed replay, non-bitwise, distribution equivalence unconfirmed;
   raw report hiện là user-reported. Không viết receipt reconstructed như log
   contemporaneous. Chỉ lập kế hoạch/cost cho full; full_launch_allowed=false
   cho đến manifest mới được review và chỉ thị launch riêng. Không gọi Modal.
