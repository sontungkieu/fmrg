Bạn là agent trên node công ty, tiếp tục chuẩn bị FMRG full-paper.
CHƯA LAUNCH FULL, không rerun shortlist hoặc gọi Modal. Không can thiệp canary
ensemble/job đang chạy; thu receipt hiện hành và chỉ chạy bước kế tiếp khi GPU
allocation/queue hiện có cho phép. Không tự mở allocation mới.

Nguồn chuẩn (handoff mới thay thế các đoạn GenEval/VLM/splits của bản29/10):
- Handoff HF immutable: https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/3bd3a35d8226b8353a1a6c23d13303618cdce683/full-paper/20261101/PREPARATION_HANDOFF.md
- SHA256 handoff: ad3c06b2de3249e47a961dffc4041a6bf2a190912164d9811386401cb8ddc743
- Publication manifest: https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/a3bc850ce0a5e90ca1f41adf6275cfa5705cb2f3/full-paper/20261101/publication.json
- SHA256 publication: 54f9b8639586f3cb5299f09b9f84c5fe6e8bbe85e83ffdae7d16f6206fe9ac34
- GitHub: https://github.com/sontungkieu/fmrg ; deployment/b200/geneval/.
- Scientific source frozen: be485ba40f1d1c163b95ad61112aca6baa13ed22.

1. Đọc AGENTS/runbook và handoff qua proxy HF được phép, verify SHA. Resolve
   layout fmrg_all, storage, proxy, allocation/job/deadline/GPU UUID từ evidence
   hiện hành. Giữ CUDA_VISIBLE_DEVICES và manager/queue đang vận hành. Không log
   credentials/proxy có auth, không bật set -x. Giữ attempts/logs cũ.

2. Tái dùng core /opt/fmrg cùng receipt/hash đã đạt. Theo đúng lệnh trong handoff,
   tải toolkit và nhóm runtime/reconstructed trực tiếp shared storage qua HF
   proxy, verify full SHA. Nhóm vlm thêm16.6GB; nhóm measurements là gói riêng,
   chưa cần cho GenEval canary. Không pip/conda/compile/GitHub download trên node.
   Skywork access đã giải quyết và weights public; paper reward adapter chưa đạt.

3. Restore GenEval overlay162MB vào /opt/fmrg-geneval theo helper, giữ core venv.
   Nếu prefix tồn tại thì verify/reuse, không restore đè, không sudo tự động.
   Dùng weights/source extras29/10 đã verify; basename detector thực tế không có
   timestamp như lệnh sai trong handoff cũ. Giữ OpenAI CLIP loader/QuickGELU.

4. Trong allocation một B200 đang được cấp, chạy canary.py đúng lệnh/timeout180s
   và output mới của handoff. TCP bị chặn trong canary. Thu JSON/log/exitcode,
   allocation/job/GPU UUID/driver, full identities, elapsed/VRAM. Modal native
   MSDA/NMS + R50/Swin-S/color technical PASS là reference; company phải có
   receipt riêng. Import-only, timeout hoặc thiếu JSON không được ghi PASS.
   R50 là paper primary, Swin-S là official-default control, giữ score tách biệt.
   Không sinh ảnh FLUX, không chạy553prompts×4seeds hoặc full inverse matrix.

5. Split AFHQ-Dog/FFHQ là reconstructed được user duyệt riêng: mỗi dataset
   50tuning+1000eval, tổng2100ảnh, có hashes/preprocessing/FFHQ attribution.
   Giữ nhãn reconstructed, không gọi author-exact hoặc chọn lại split theo metric.
   Nếu stage observations thì kiểm manifest/operator-state/tensor hashes; chưa
   có qualified runner adapter nạp frozen y. Released CLI hiện tự tạo observations;
   không gọi generator mặc định rồi tuyên bố đã sử dụng frozen inputs.

6. Đọc parent PAPER_SOURCE_AUDIT/CONTRACT và CONTRACT_UPDATE mới. Hai track
   released_code_v1 và paper_formula_v1 khác loss/Eweight/early-stop/Adam-J;
   paper adapter chưa qualify. Skywork cần differentiable Yes/No-logit reward,
   không scalar valuehead; renoiseconstantc vẫn chưa chốt. Style16refs+48paired
   prompts, tuning objective/tie-break/baselines, compute instrumentation và
   full allocation/deadline vẫn là blockers. Draft tuning không phải author budget.
   Không đổi sampler frozen/preprocessing để cải thiện số đo.

7. Trả report bỏ secrets: nhóm tải/hash/revisions, core/overlay/source/contract,
   allocation/job/GPU UUID, JSON/log/exitcode/time/VRAM từng canary, ensemble receipt
   nếu đã có, và từng blocker kèm evidence path/bước xử lý. Missing/failed giữ rõ,
   không zero-fill. Screening replay non-bitwise là báo cáo trước; chưa có raw
   receipt mới thì không nâng thành kết luận thống kê/efficacy. full_launch_allowed
   luôn false; full execution chỉ sau manifest được review và chỉ thị launch riêng.
