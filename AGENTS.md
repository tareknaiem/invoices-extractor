# AGENTS.md — Invoices Extractor

ملخّص مرجعي لمجهر المشروع (للعمل على الكود بالذكاء الاصطناعي والبشر معًا).
القواعد غير القابلة للتفاوض موثّقة في `constitution.md`؛ هذا الملف يشرح **البنية والآلية**.

## نظرة عامة

خط معالجة (Pipeline) محلي لاستخراج فواتير الفنادق/الرحلات من PDF وإخراجها كملف
إكسل موحّد:

```
PDF → extractor (نص أو OCR) → formatter (الهيكل القياسي) → exporter (إكسل حيّ) → Streamlit UI
```

نقطة الدخول الموحّدة: `main.run(pdf_path, supplier_key=None, output_path=None,
engine="auto", ocr_engine=None, ocr_resolution=None, ocr_max_pages=None)` التي
تعيد `(output_path, formatted_df, summary_dict, supplier_key)`.

## التقنيات والمكتبات (من requirements.txt)

| المكتبة | الإصدار | الدور |
|---|---|---|
| **Python** | 3.14 (بيئة `venv/`) | لغة المشروع؛ ملاحظة: عجلات Paddle تدعم 3.9–3.13 فقط، لذا OCR قد يكون غير متاح على 3.14 |
| **Streamlit** | 1.64.0 | واجهة الويب في `app.py` (`streamlit run app.py`) |
| **pdfplumber** | 0.11.10 | استخراج النصوص والجداول من PDF النصي + تحويل الصفحات إلى صور للـ OCR عبر محرك pypdfium2 المدمج |
| **PaddleOCR / PaddlePaddle** | 3.7.0 / 3.3.1 | محرك OCR العربي (`lang='ar'`) — اختياري، يُستورد بشكل كسول (lazy) ويُبلّغ `OcrUnavailableError` إن غاب |
| **RapidFuzz** | 3.14.6 | مطابقة أعمدة سريعة (Fuzzy) في `formatter`؛ يسقط تلقائيًا إلى `difflib` المكتبة القياسية إن لم يُثبَّت |
| **OpenPyXL** | 3.1.5 | كتابة ملف الإكسل المنسّق (صيغ حيّة + تنسيقات) |
| pandas / numpy / PyYAML | 3.0.6 / 2.5.3 / 6.0.3 | جداول البيانات، مصفوفات OCR، قراءة قواعد المورّدين |

## خريطة الملفات الرئيسية

| الملف | الوظيفة بدقة |
|---|---|
| **`main.py`** | المنسّق (Orchestrator): دالة `run()` تمزج المسارين وتكتب الإكسل. `_resolve_engine()` يختار محرك الاستخراج. `ENGINES = ("auto", "text", "ocr")`. |
| **`app.py`** | واجهة Streamlit: رفع PDF متعدد، شريط جانبي (وضع المعالجة: Auto/Text/OCR، اختيار المورّد يدويًا أو Auto-detect)، تنفيذ عبر `process_pdf()` → `main.run()`، عرض `st.dataframe` + أزرار تحميل `st.download_button` + رسائل نجاح/خطأ لكل ملف. يعمل بذاكرة مؤقتة (`tempfile.TemporaryDirectory`). |
| **`extractor_pdf.py`** | محرك النص: استخراج الجداول بـ pdfplumber، كشف صف العناوين (`detect_header_row`)، تنظيف الخلايا والعملات (`_clean_currency`)، تصحيح عربي الترتيب البصري (`_fix_arabic`)، فصل صفوف/جداول الملخّص (`_is_summary_table`, `_drop_summary_rows`)، كشف ضوضاء `(cid:NN)` (`has_cid_garbage`)، ومحرك الـ **Smart Fallback** للمورّدين المجهولين (`smart_extract_tables` / `smart_fallback_extract`). ثابت `DYNAMIC_SUPPLIER_KEY = "auto_detected"`. |
| **`extractor_ocr.py`** | محرك البصري: تحويل الصفحات لصور (`render_pdf_pages`, افتراضي 300 DPI)، تغليف `PaddleOcrEngine` ( كسول، يدعم API ‏v3/v2 مع تدرّج في التوافق)، تطبيع مخرجات Paddle (`_parse_paddle_result`)، إعادة بناء جدول من صناديق النص (`lines_to_rows` → `column_bands` → `lines_to_table`)، وقطّاع التوجيه `should_use_ocr()`. `ocr_dependency_status()` / `is_ocr_available()` لحالة المحرك. |
| **`formatter.py`** | المنسّق: تحويل لأي إطار إلى الهيكل القياسي `standard_columns`. مساران: `format_transactions()` بملف المورّد (`columns_mapping`) أو `format_dynamic()` بالمطابقة الضبابية (`build_column_mapping` ← `best_standard_match` + `detect_header_currency`). كذلك: `detect_supplier()`, تطبيع التواريخ إلى `YYYY-MM-DD`, أعداد `Adult/Child/Inf` صحيحة، طيّ زائد الأعمدة في `Notes`، ومرادفات متعددة اللغات `STANDARD_COLUMN_ALIASES` (عربية + معكوسة). عتبة القبول `FUZZY_MATCH_THRESHOLD = 0.80`. |
| **`exporter.py`** | المُصدّر: `export_to_excel(df, output_path, summary, sheet_name="Invoices", total_label="Total")`. يبني: كتلة أسعار الصرف القابلة للتحرير (صفوف 1–3؛ `$B$2` USD=50، `$D$2` EUR=55، `$F$2` GBP=65)، معادلات حيّة في `EGP Converter` لكل صف (`=N(EGP)+N(USD)*$B$2+…`)، صف Subtotal بـ `=SUM(...)` لكل عمود مالي، تنسيقات/حدود/AutoFilter/تجميد، وأوراق ملخّص اختيارية. `HEADER_ROW = 4`. |
| **`suppliers_rules.yaml`** | الإعداد: `standard_columns` (14 عمودًا: EGP Converter, EGP, GBP, Euros, USD, V.NO, Hotel, Date, Adult, Child, Inf, Total, Notes, Net) + 6 مورّدين (`egypt_gate`, `sharm_gate`, `egypt_gate_bebo`, `egypt_gate_sinai`, `sharm_gate_mohamed`, `smile_safari`) كل بـ `identifiers` و`columns_mapping` و`notes_columns`. |
| **`tests/test_fallback_extractor.py`** | 17 اختبارًا: المطابكة الضبابية والعملات، كشف صف العناوين، الـ Smart Fallback (بيانات عربية)، صيغ `EGP Converter` و`SUM` في التصدير، و`main.run()` مع مورّد مجهول. يعمل مباشرة بدون pytest. |
| **`tests/test_ocr_pipeline.py`** | 24 اختبارًا: كشف/تنظيف `(cid:)`، قطّاع التوجيه، الرسم بياني للصفحات، تحليل مخرجات Paddle v2/v3، بناء الجدول من الصناديق، ونهاية-لنهاية عبر محرك وهمي (`_FakeEngine`) يؤكد بقاء صيغ `SUM` + `EGP Converter`. يعمل مباشرة بدون pytest. |
| **`constitution.md`** | قواعد العمل الإلزامية (تكامل، أمان، عربي/إكسل، تحقق 100%). |
| **`requirements.txt`** | الاعتمادات المقفولة، مع ملاحظات شرطية لـ OCR على Python ≥3.14. |
| **`.devcontainer/devcontainer.json`** | حاوية التطوير. |

**تشغيل:** `streamlit run app.py` للواجهة · `python main.py <pdf> [supplier] [engine]`
للسطر · الاختبارات: `python -m pytest tests/ -v` أو مباشرة
`python tests/test_fallback_extractor.py` و`python tests/test_ocr_pipeline.py`
(كل ملف يضمّ مشغّلًا ذاتيًا `_run_all()` لا يحتاج pytest).

---

## آليّة التوجيه: النص مقابل OCR (Fallback) وربط المُصدّر

### 1. اختيار المحرك — `main._resolve_engine(engine, pdf_path)`

المعامل `engine` أحد `("auto", "text", "ocr")` (الافتراضي `"auto"`، وفي الواجهة
يتحوّل عبر `st.radio` في الشريط الجانبي):

| الحالة | القرار |
|---|---|
| `"text"` | مسار النص دائمًا (لا OCR) |
| `"ocr"` | مسار OCR دائمًا |
| `"auto"` | يقرأ `extractor_pdf.get_raw_pdf_text()` (خام — يحتفظ بعلامات `(cid:NN)`) ثم يسأل `extractor_ocr.should_use_ocr(text, pdf_path)` |
| أي قيمة أخرى | `ValueError` |

### 2. قرار الـ Fallback — `extractor_ocr.should_use_ocr()`

تعود `True` (أي: اذهب إلى OCR) عند تحقق أي شرط:

1. **ضوضاء ترميز:** النص يحتوي `(cid:NN)` (`extractor_pdf.has_cid_garbage`) —
   خطوط PDF بلا خريطة `ToUnicode` (شائع مع العربية) تُستخرج كرموز بديلة غير مقروءة.
2. **نص فارغ/قصير:** `len(text) < MIN_CHARS_PER_PAGE × عدد الصفحات` حيث
   `MIN_CHARS_PER_PAGE = 40` — أي PDF ممسوح ضوئيًا بلا طبقة نص.

> **تحذير موثّق في الكود:** يجب تمرير النص الخام عبر `get_raw_pdf_text`؛ الدالة
> `get_pdf_text` تحذف علامات `(cid:)` فتُعطّل الكشف تلقائيًا.

### 3. المساران بعد القرار (داخل `main.run`)

**أ) مسار النص (Text):**
```
get_pdf_text(pdf) ──► formatter.detect_supplier(text, rules)
   │
   ├─ مورّد معروف ──► extractor_pdf.extract_tables_from_pdf()   # جداول + فصل الملخّص
   │                   └─► formatter.format_transactions(df, supplier_key, rules)  # profile ثابت
   │
   └─ مجهول (None/غير صالح) ──► extractor_pdf.smart_fallback_extract()           # Smart Fallback
                                 (كشف عناوين + فصل الملخّص + formatter.format_dynamic)
                                 └─► supplier_key = DYNAMIC_SUPPLIER_KEY ("auto_detected")
```

**ب) مسار البصري (OCR):**
```
extractor_ocr.extract_transactions_from_pdf(pdf, rules, engine, resolution, max_pages)
   │   render_pdf_pages (300 DPI) ─► PaddleOcrEngine.recognize (lang='ar')
   │   ─► lines_to_rows / column_bands / lines_to_table   # إعادة بناء شبكة الجدول
   │   ─► extractor_pdf.tables_to_transactions(fix_arabic=False)  # عربي أصلًا بالمنطق السليم
   │   (يُعاد أيضًا ocr_text كامل)
   ├─► formatter.detect_supplier(ocr_text, rules)     # إن لم يُحدَّد يدويًا
   └─► formatter.format_dynamic(raw, standard_columns)  # دائمًا: رؤوس OCR ضوضائية
        └─► إن المورّد غير معروف: supplier_key = DYNAMIC_SUPPLIER_KEY
```

**ملاحظات تكاملية:**
- كلا المسارين ينتهي بنفس الشكل: `pd.DataFrame` بأعمدة `standard_columns` بالترتيب
  + `dict` ملخّص، ثم يمرّ من `exporter.export_to_excel` — لذلك الفرق بين المسارين
  في **الاستخراج فقط**، لا في التنسيق أو التصدير.
- OCR يستخدم `fix_arabic=False` لأن Paddle يعيد النص بالترتيب المنطقي أصلًا،
  بينما مسار النص يستخدم `fix_arabic=True` لتقليل النص المخزّن بالترتيب البصري.
- المورّد غير المعروف في المسار النصي يعمل عبر Smart Fallback، أما OCR فلا
  يستخدم ملفات المورّدين إطلاقًا (التوافق الضبابي فقط).
- غياب PaddleOCR لا يوقف مسار النص: `OcrUnavailableError` تحمل رسالة إرشادية
  تُلتقط في `app.py` وتظهر كخطأ في الواجهة.

### 4. الربط بمُصدّر الإكسل — `exporter.export_to_excel`

`main.run` تستدعي في الأخير (للمسارين معًا):
```python
output_path = exporter.export_to_excel(formatted, output_path, summary=summary)
```
وبداخلها:

1. **الكتابة الأولية:** `DataFrame.to_excel` بمحرّك openpyxl — البيانات تبدأ تحت
   كتلة أسعار الصرف، أي من `HEADER_ROW = 4` (الصف 4 = العناوين، 5+ = البيانات).
2. **كتلة أسعار الصرف** (`_write_rate_block`): صفوف 1–3؛ خلايا قابلة للتحرير
   `$B$2`(USD=50), `$D$2`(EUR=55), `$F$2`(GBP=65).
3. **صيغ التحويل الحية** (`_write_converter_formulas`): لكل صف في عمود
   `EGP Converter`:
   `=N(EGP)+N(USD)*$B$2+N(Euros)*$D$2+N(GBP)*$F$2` — دالة `N()` تمنع كسر المعادلة
   عند الخلايا الفارغة/النصية.
4. **صف الإجمالي** (`_write_total_row`, عند وجود صفوف بيانات): تسمية نصية
   (`Total` أو `الاجمالي` عبر `total_label`) في عمود نصي (يفضّل Hotel/Date/V.NO/Notes)
   و`=SUM(<col><data_start>:<col><data_end>)` لكل عمود مالي فوق نطاق البيانات
   الفعلي حصريًا (لا إحالة دائرية).
5. **التنسيق** (`_style_sheet` + `_style_total_row`): تنسيق `#,##0.00` للعملات و`0`
   للأعداد، AutoFilter يغطي البيانات فقط (لا صف الإجمالي)، تجميد صف العناوين،
   حدود/تعبئة، تجاور عرض الأعمدة، وورقة `Summary` منسّقة عند توفر `summary`.

أي ميزة جديدة تمرّ بهذا المسار يجب أن تبقى حيّة في الإكسل (صيغ لا قيم ثابتة)
وأن يؤكد اختباران وجودها: `test_fallback_export_has_live_egp_formulas` و
`test_export_appends_total_row_with_sum_formulas` (وكذلك مقابلَيهما في مسار OCR).

