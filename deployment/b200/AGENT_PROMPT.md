Bạn là agent đang làm việc trên node công ty. Hãy triển khai FMRG theo thứ tự:
đọc handoff HF → setup runtime/assets → canary B200 → replay shortlist → chốt
contract và kế hoạch reproduce full paper. Tự làm các bước an toàn đã được phép;
chỉ hỏi đúng thông tin không lấy được từ môi trường/receipt hiện hành.

Nguồn chuẩn:
- Handoff immutable: https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/6352766e35c892ec51d70dcf79e5c4968a15de7e/HANDOFF.md
- SHA256 handoff: 06bebbc78965094e187f341dd3bd81fee12c51a24190d713ca7b04426a1b305c
- Repo code: https://github.com/sontungkieu/fmrg ; thư mục deployment/b200/.
- Scientific source trong runtime: be485ba40f1d1c163b95ad61112aca6baa13ed22.

1. Đọc AGENTS/runbook hiện hành và handoff trước. Lấy hostname, allocation đúng
   một B200, GPU UUID, queue/manager/receipt, deadline, shared storage và proxy HF
   từ môi trường thực tế. Không đoán từ hostname/GPU của campaign cũ; số GPU nhìn
   thấy không chứng minh quyền sử dụng. Giữ CUDA_VISIBLE_DEVICES scheduler cấp.
   Dùng manager đang vận hành, không tạo manager/state trùng hoặc sửa SQL.

2. Dùng layout project đã duyệt: fmrg_all/fmrg cho checkout nếu cần, tài nguyên
   riêng trong fmrg_all/{tmp,models,logs,runs}; giữ model dùng chung tại chỗ.
   Xác minh root được phép, Linux x86_64, driver CUDA12.8, libGL/glib, quyền ghi
   /opt và khoảng100GiB trống. Không tự move dữ liệu hay cài/đổi driver hệ thống.
   Nếu runtime/data đã có, kiểm identity/receipt để tái dùng, không coi tồn tại
   là PASS và không overwrite /opt/fmrg hoặc output của attempt trước.

3. Tải handoff/helper/manifest qua proxy HF hiện hành và verify SHA trước sử
   dụng. Dùng đúng lệnh setup và immutable URLs trong handoff. Các file public
   không cần HF token; không log token/proxy có auth, không bật set -x. Node chỉ
   tải qua HF/proxy; không fallback GitHub Raw, PyPI, git clone hoặc cài dependency
   mới trên node. GitHub dùng làm nguồn code/provenance, không phải đường tải bắt
   buộc. Runtime/model lớn tải trực tiếp trên shared storage, không qua laptop.

4. Đọc helper trước khi chạy: helper vừa tải/restore vừa chạy bốn gates. Tách
   chuẩn bị CPU khỏi GPU nếu queue hỗ trợ, giữ nguyên checksum contracts; không
   bịa option CLI. Phần GPU phải nằm trong allocation/job được phép với timeout
   hữu hạn. Restore prefix đúng /opt/fmrg, data riêng; lấy data_root/pilot_root từ
   publication-used.json. Không pip/conda/uv sync hoặc rebuild runtime chỉ vì
   checkout/source/config đổi. Sampling dùng /opt/fmrg/python-fmrg.sh,
   FMRG_SOURCE_ROOT=/opt/fmrg/source và HF_HUB_OFFLINE=1.

5. Canary B200 gồm imports → kernel → tiny → full (256px unguided/E/J). Thu JSON,
   log, exit code, GPU/driver, thời gian và peak memory; yêu cầu cả bốn gate PASS
   và setup-summary TECHNICAL_PASS. Nếu fail, giữ traceback đầu tiên và receipt,
   sửa nguyên nhân trong phạm vi được phép rồi tạo attempt mới, không retry mù
   hoặc bỏ gate. Canary/import pass không phải kết quả khoa học hay chứng nhận
   full reward ensemble. J không có bảo đảm bitwise equality.

6. Sau canary PASS, replay offline theo source/config frozen từ HF, qua queue
   với dependency success: G (CPU) 288 configs/8 oracle checks; I72, J96, R24
   unique cells. Dùng seed/input/measurement/hyperparameters cũ, output mới mỗi
   attempt; giữ nguyên deadline và ghi resolved config/hash. I/J/R runner chạy
   toàn bộ grid, không có cờ one-cell; không gọi chúng như canary ngắn. R là
   CLIP-only512px, không phải ensemble của paper. K3 có extra selection cost.
   Không gọi các launcher Modal hoặc tiêu thêm ngân sách Modal.

7. Tải reference raw results từ link HF trong handoff; so sáu measurement hashes,
   expected/actual unique cells, summary COMPLETE, finite metrics, gradient/NFE
   counters và ảnh. Định nghĩa tolerance trước khi so, báo chênh PSNR/SSIM/CLIP,
   latency/VRAM; không đổi strength/loss để ép khớp và không coi exit0 là đủ.
   Counter R autograd_calls không đếm Tensor.backward; sửa instrumentation bằng
   overlay/manifest mới nếu cần đo đầy đủ, không tráo nó với replay nguyên bản.

8. Trước full paper, chốt và lưu: AFHQ-Dog/FFHQ tuning/eval splits, hash và
   preprocessing; squared-pixel so với norm/latent reward; actual time grid và
   early-stop; full reward/evaluator/LPIPS/FID/GenEval assets; tuning budget,
   seeds, actual compute và allocation/deadline. Không xem 3 ảnh hay2 prompts là
   full benchmark, không áp headline3NFE cho mọi setting. Chưa launch full matrix
   khi các contract/tài nguyên này chưa chốt. Lập kế hoạch đầy đủ với acceptance
   criteria, ước lượng tài nguyên và blocker còn lại trước khi mở rộng.

9. Kết thúc bằng receipt/report bỏ secrets: source/runtime/assets/HF revisions,
   host/GPU/allocation/queue job IDs, resolved paths/config, trạng thái từng gate,
   expected/actual outputs, metrics/diffs/failures, thời gian/VRAM, evidence paths
   và bước tiếp theo. Phân biệt setup/canary PASS, replay screening và full-paper
   reproduction. Commit thay đổi code đúng phạm vi; không sửa job đang chạy hoặc
   coi quyền setup/replay là quyền tự mở rộng allocation/ngân sách.
