# Day 16 - Track 2: Cloud Infrastructure for AI  
## Báo cáo Benchmark phương án CPU thay thế GPU

**Họ tên:** Lê Hoàng Nam
**Mã Sinh Viên** 2A202600965
**Mục tiêu:** Triển khai hạ tầng cloud trên AWS, chạy benchmark Machine Learning trên CPU node và kiểm tra chi phí vận hành.  
**Dataset:** Credit Card Fraud Detection (`mlg-ulb/creditcardfraud`)  
**Model:** LightGBM Classifier  
**Instance benchmark:** `r5.2xlarge` CPU node  
**Region:** `us-east-1`  
**Phương án:** CPU + LightGBM thay thế GPU

---

## 1. Kiến trúc triển khai

Hạ tầng được triển khai bằng Terraform trên AWS, gồm các thành phần chính:

| Thành phần | Vai trò |
|---|---|
| VPC | Mạng riêng cho toàn bộ hạ tầng |
| Public Subnet | Chứa Bastion Host và Load Balancer |
| Private Subnet | Chứa CPU node chạy benchmark |
| Bastion Host | Máy trung gian để SSH vào private CPU node |
| CPU Node `r5.2xlarge` | Chạy benchmark LightGBM |
| NAT Gateway | Cho private node truy cập Internet để tải package/dataset |
| Application Load Balancer | Endpoint HTTP phục vụ kiểm thử hạ tầng |
| Security Groups | Kiểm soát truy cập SSH/HTTP nội bộ |

Luồng truy cập SSH:

```text
Local machine -> Bastion Public IP -> CPU Private IP
```

---

## 2. Lý do dùng CPU thay GPU

Trong bài lab này, nhóm sử dụng phương án **CPU + LightGBM** thay cho GPU vì:

1. Tài khoản AWS mới có thể bị giới hạn quota hoặc cần xác minh khi tạo GPU instance.
2. GPU instance thường có chi phí cao hơn, dễ phát sinh phí nếu quên xoá tài nguyên.
3. Bài toán Credit Card Fraud Detection là dữ liệu dạng bảng, phù hợp với LightGBM.
4. LightGBM chạy rất hiệu quả trên CPU, không bắt buộc cần GPU để đạt kết quả tốt.
5. CPU node vẫn đủ để benchmark training time, AUC, inference latency và throughput.

---

## 3. Dataset

Dataset được tải từ Kaggle:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/
```

Dataset gồm khoảng **284,807 giao dịch thẻ tín dụng**, trong đó nhãn `Class = 1` là giao dịch gian lận và `Class = 0` là giao dịch bình thường. Đây là bài toán mất cân bằng dữ liệu, nên metric quan trọng nhất không chỉ là Accuracy mà còn là AUC-ROC, Precision, Recall và F1-Score.

---

## 4. Cấu hình benchmark

Các thư viện sử dụng:

```bash
pip3 install lightgbm scikit-learn pandas numpy kaggle
```

Model sử dụng:

```python
LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

Dữ liệu được chia train/test theo tỉ lệ:

```text
Train: 80%
Test: 20%
Stratify theo nhãn Class
```

---

## 5. Kết quả benchmark

Kết quả chạy `python3 benchmark.py` trên CPU node `r5.2xlarge`:

| Metric | Kết quả |
|---|---:|
| Thời gian load data | `1.6067 sec` |
| Thời gian training | `12.5747 sec` |
| Best iteration | `1000` |
| AUC-ROC | `0.976687` |
| Accuracy | `0.999579` |
| F1-Score | `0.872340` |
| Precision | `0.911111` |
| Recall | `0.836735` |
| Inference latency 1 row | `1.319647 ms` |
| Inference throughput 1000 rows | `52847.62 rows/sec` |

File kết quả được lưu ra:

```text
metrics.json
metrics.csv
```

---

## 6. Nhận xét kết quả

Kết quả benchmark cho thấy CPU node `r5.2xlarge` đủ mạnh để xử lý bài toán tabular ML với LightGBM. Thời gian training khoảng **12.57 giây**, tương đối nhanh với dataset hơn 280 nghìn dòng. AUC-ROC đạt **0.976687**, cho thấy mô hình phân biệt tốt giữa giao dịch gian lận và giao dịch bình thường. Accuracy rất cao (**0.999579**), tuy nhiên vì dataset mất cân bằng nên cần xem thêm Precision, Recall và F1-Score. F1-Score đạt **0.872340**, Precision đạt **0.911111**, Recall đạt **0.836735**, cho thấy mô hình phát hiện fraud khá tốt nhưng vẫn còn một số giao dịch gian lận bị bỏ sót.

Về inference, latency cho một dòng chỉ khoảng **1.32 ms**, throughput đạt khoảng **52,847 rows/sec**, phù hợp cho demo endpoint hoặc xử lý batch nhỏ. Với workload dạng bảng, CPU + LightGBM là lựa chọn hợp lý hơn GPU vì chi phí thấp hơn, triển khai đơn giản hơn và vẫn đạt hiệu năng tốt.

---

## 7. Kiểm tra chi phí

Sau khi triển khai hạ tầng, nhóm kiểm tra AWS Billing/Bills. Tại thời điểm kiểm tra, AWS Billing có thể chưa cập nhật chi phí thực tế do dữ liệu billing thường có độ trễ vài giờ đến 24 giờ. Vì vậy, báo cáo sử dụng bảng ước tính chi phí theo giờ trong region `us-east-1`.

Ước tính chi phí 1 giờ:

| Dịch vụ | Instance/Loại | Chi phí/giờ ước tính |
|---|---|---:|
| EC2 CPU Node | `r5.2xlarge` | `~$0.504` |
| EC2 Bastion | `t3.micro` | `~$0.010` |
| NAT Gateway | Mỗi AZ | `~$0.045 + data` |
| Application Load Balancer | ALB | `~$0.008` |
| **Tổng ước tính** |  | **~$0.57/giờ** |

Ghi chú: Chi phí thực tế có thể khác do data transfer, thời gian tài nguyên chạy thực tế và độ trễ cập nhật của AWS Billing.

---

## 8. Các minh chứng cần nộp

Các file/screenshot cần đưa vào repository:

```text
benchmark.py
metrics.json
metrics.csv
terraform/
screenshots/benchmark_terminal.png
screenshots/aws_billing.png
report_cpu_benchmark.md
```

Trong đó:

- `benchmark.py`: mã nguồn chạy benchmark.
- `metrics.json`: kết quả benchmark dạng JSON.
- `metrics.csv`: kết quả benchmark dạng CSV.
- `terraform/`: mã nguồn Terraform đã chỉnh sửa để dùng CPU node `r5.2xlarge`.
- `screenshots/benchmark_terminal.png`: ảnh terminal chạy `python3 benchmark.py`.
- `screenshots/aws_billing.png`: ảnh AWS Billing/Bills hoặc Cost Explorer.
- `report_cpu_benchmark.md`: báo cáo ngắn này.

---

## 9. Lưu ý bảo mật và dọn dẹp tài nguyên

Không được commit các file nhạy cảm lên GitHub:

```text
lab-key
lab-key.pub
kaggle.json
*.tfstate
*.tfstate.backup
.terraform/
.env
```

Sau khi hoàn thành benchmark và chụp đủ minh chứng, cần xoá hạ tầng để tránh phát sinh chi phí:

```bash
terraform destroy
```

Sau đó nhập:

```text
yes
```

---

## 10. Kết luận

Bài lab đã triển khai thành công hạ tầng AWS bằng Terraform, kết nối vào CPU node thông qua Bastion Host, tải dataset từ Kaggle và chạy benchmark LightGBM. Phương án CPU thay GPU phù hợp với bài toán dữ liệu bảng vì chi phí thấp, triển khai nhanh và vẫn đạt AUC-ROC cao. Kết quả benchmark cho thấy `r5.2xlarge` xử lý tốt workload Credit Card Fraud Detection với training time ngắn và inference throughput cao. Đây là phương án hợp lý khi tài khoản AWS chưa đủ quota GPU hoặc cần tối ưu chi phí cho bài lab.
