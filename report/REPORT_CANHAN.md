# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thúy Linh
**Nhóm:** K4-L3B
**Ngày:** 20/09/2026

> Nộp 1 bản / sinh viên. Phần nhóm được tổng hợp chung trong REPORT_NHOM.md.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao có nghĩa là hai vector hướng gần nhau trong không gian embedding, nên chúng mang ý nghĩa tương đồng dù không nhất thiết cùng từ khóa.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Học sinh đang học Python để làm dự án AI."
- Câu B: "Người dùng học Python để xây dựng ứng dụng trí tuệ nhân tạo."
- Tại sao tương đồng: Cả hai chủ yếu nói về cùng một chủ đề là Python và AI, dù từ ngữ khác nhau.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Con mèo ngủ trên ghế."
- Câu B: "Máy bay đang cất cánh ở sân bay."
- Tại sao khác: Nội dung của hai câu không cùng lĩnh vực và không chia sẻ ý nghĩa chính.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine similarity tập trung vào hướng của vector, tức là ý nghĩa ngữ nghĩa, còn Euclidean distance nhấn mạnh độ lớn tuyệt đối của vector. Với văn bản, hướng biểu diễn ý nghĩa thường quan trọng hơn so với khoảng cách tuyệt đối.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: số chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.111...) = 23 chunks.
>
> **Đáp án:** 23 chunks.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, bước dịch chuyển giảm từ 450 xuống 400, nên số chunk sẽ tăng lên: ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25 chunks.
>
> Độ chồng chéo giúp duy trì ngữ cảnh giữa các chunk liền kề và giảm mất thông tin ở các ranh giới chunk, nhưng quá nhiều overlap làm tăng độ trùng lặp và giảm hiệu quả lưu trữ.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi dùng regex tách theo ranh giới câu như `(?<=[.!?])\s+|\n+` để chia văn bản thành các câu, sau đó gom nhóm theo `max_sentences_per_chunk` và nối bằng khoảng trắng. Với trường hợp văn bản rỗng hoặc chỉ có khoảng trắng, tôi trả về danh sách rỗng để tránh lỗi.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán hoạt động theo kiểu đệ quy: bắt đầu với các dấu phân cách ưu tiên như xuống dòng, dấu chấm, khoảng trắng; nếu đoạn văn bản quá dài, ta chia theo separator gần nhất và tiếp tục đệ quy trên các đoạn con. Trường hợp cơ sở là khi độ dài đoạn hiện tại nhỏ hơn hoặc bằng `chunk_size`, ta dừng lại và trả về đoạn đó.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Mỗi tài liệu được nhúng bằng hàm embedding được truyền vào, rồi lưu dưới dạng bản ghi chứa `content`, `metadata`, `embedding` và `doc_id`. Khi tìm kiếm, tôi tính tích vô hướng giữa vector truy vấn và từng vector trong kho để xếp hạng độ tương đồng.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Tôi lọc metadata trước khi tính toán tương đồng để loại bỏ các chunk không phù hợp với điều kiện truy vấn. Với `delete_document`, tôi xóa tất cả các bản ghi có `metadata['doc_id'] == doc_id` để đảm bảo toàn bộ các chunk của một tài liệu bị xóa đồng bộ.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Tác tử đầu tiên gọi `store.search(question, top_k)` để lấy các chunk liên quan nhất, sau đó xây dựng prompt chứa câu hỏi và ngữ cảnh. Cấu trúc prompt được thiết kế đơn giản nhưng rõ ràng: đặt câu hỏi lên đầu, sau đó là các chunk context và yêu cầu trả lời chỉ dựa trên context đó.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```bash
cd /d/K4-L3B-Day07-Nguyen-Thuy-Linh-2A202602497
pytest tests/test_solution.py -q
```

Kết quả thực tế sau khi hoàn thiện:
- `42/42` tests pass
- Tất cả các bài kiểm thử liên quan đến chunking, vector store và knowledge base đều đạt.

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Python is a programming language." | "Python is used for software development." | cao | cao | Có |
| 2 | "The cat sleeps on the sofa." | "A dog barks in the yard." | thấp | thấp | Có |
| 3 | "Machine learning learns from data." | "Models improve using training examples." | cao | cao | Có |
| 4 | "I like coffee." | "The sky is blue." | thấp | thấp | Có |
| 5 | "Vector databases store embeddings." | "Embeddings are used for similarity search." | cao | cao | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Điều đáng ngạc nhiên nhất là hai câu không cùng từ khóa nhưng vẫn có độ tương tự cao khi chúng mô tả cùng một khái niệm. Điều này cho thấy embedding phản ánh ý nghĩa ngữ nghĩa, không chỉ khớp chữ.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | "Python là ngôn ngữ lập trình gì?" | "Python is a high-level programming language." | 0.88 | Có | Trả lời Python là ngôn ngữ lập trình bậc cao, dễ học và phổ biến. |
| 2 | "Machine learning hoạt động như thế nào?" | "Machine learning uses algorithms to learn from data." | 0.81 | Có | Tóm tắt việc mô hình học từ dữ liệu để cải thiện dự đoán. |
| 3 | "Vector database dùng để làm gì?" | "Vector databases store embeddings for similarity search." | 0.86 | Có | Giải thích về lưu trữ embedding và tìm kiếm tương đồng. |
| 4 | "Lợi ích của embedding là gì?" | "Embeddings convert meaning into numeric vectors." | 0.74 | Có | Nói rằng embedding giúp biểu diễn ngữ nghĩa trong không gian vector. |
| 5 | "Tại sao similarity search quan trọng?" | "Similarity search helps retrieve semantically related information." | 0.79 | Có | Giải thích rằng tìm kiếm tương đồng giúp lấy thông tin phù hợp về ý nghĩa. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Tôi học được rằng cùng một bộ dữ liệu nhưng chiến lược chia chunk khác nhau có thể mang lại độ chính xác truy xuất khác biệt rõ rệt. Một số câu hỏi cần metadata để lọc đúng trường hợp, còn một số câu hỏi khác lại cần chunk có tính liên tục hơn để giữ ngữ cảnh.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
