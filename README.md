# Project 4 — Multi-Agent Compliance Research Crew

Sistem multi-agent berbasis LangGraph untuk menjawab pertanyaan kompleks seputar
tarif impor EU. Query seperti *"riset implikasi tarif baru EU untuk produk X dan
tulis ringkasan eksekutif"* dikerjakan oleh tiga agent spesialis yang
dikoordinasi satu Supervisor.

---

## Daftar Isi

1. [Arsitektur](#arsitektur)
2. [Shared State](#shared-state)
3. [LLM Config](#llm-config)
4. [Tools](#tools)
5. [Setiap Agent](#setiap-agent)
6. [Graph & Routing](#graph--routing)
7. [Alur End-to-End](#alur-end-to-end)
8. [Cara Menjalankan](#cara-menjalankan)
9. [Kelemahan Nyata & Trade-off](#kelemahan-nyata--trade-off)

---

## Arsitektur

```
User Query
    |
    v
+-------------------+
|    graph.invoke() |  <-- entry point di main.py
+-------------------+
    |
    v
+-------------+        RouteDecision (Pydantic)
| SUPERVISOR  | ----> Researcher | Analyst | Writer | FINISH
+-------------+
    ^    |
    |    +-----------> [Researcher] -> kembali ke Supervisor
    |    +-----------> [Analyst]    -> kembali ke Supervisor
    |    +-----------> [Writer]     -> kembali ke Supervisor
    |
    +--- FINISH -> END (graph berhenti)
```

Semua node berbagi satu objek `AgentState`. Setiap agent menulis ke
field-nya masing-masing; Supervisor membaca field-field itu untuk
memutuskan siapa yang dipanggil berikutnya.

---

## Shared State

**File:** `p4_multi_agent/state.py`

```python
class AgentState(TypedDict):
    messages:          Annotated[Sequence[BaseMessage], operator.add]
    query:             str   # query asli user, tidak berubah sepanjang workflow
    research_findings: str   # diisi Researcher
    analysis_results:  str   # diisi Analyst
    final_report:      str   # diisi Writer
    next:              str   # diisi Supervisor: "Researcher"|"Analyst"|"Writer"|"FINISH"
    iteration_count:   int   # bertambah 1 setiap Supervisor dipanggil
```

### Kenapa `operator.add` untuk `messages`?

Field `messages` pakai `Annotated[..., operator.add]` — artinya setiap
node yang mengembalikan `messages` akan **menambahkan** ke list yang ada,
bukan menimpa. Ini pola LangGraph v1.x untuk message accumulation.

Field lain (`research_findings`, `analysis_results`, `final_report`) adalah
string biasa — node yang mengisinya cukup `return {"research_findings": "..."}`,
dan LangGraph akan menimpa nilai sebelumnya.

---

## LLM Config

**File:** `p4_multi_agent/llm.py`

Semua agent import LLM dari satu tempat. Kalau mau ganti provider, cukup
edit file ini — tidak perlu sentuh kode agent sama sekali.

```python
fast_llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0)    # Supervisor, Researcher, Analyst
writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # Writer
```

Keduanya mengarah ke endpoint SumoPod (`https://ai.sumopod.com/v1`) yang
kompatibel dengan OpenAI SDK. API key dibaca dari `.env`:

```
SUMOPOD_API_KEY=sk-...
SUMOPOD_BASE_URL=https://ai.sumopod.com/v1
```

`temperature=0` untuk agent yang butuh output deterministik (routing,
kalkulasi). `temperature=0.3` untuk Writer agar laporan terasa natural,
bukan robotik.

---

## Tools

Tools adalah fungsi Python biasa yang didekorasi `@tool`. LLM tidak
menjalankannya langsung — LLM meminta tool tertentu dipanggil dengan
argumen tertentu, lalu LangGraph yang mengeksekusinya dan mengembalikan
hasilnya ke LLM.

### `search_compliance_docs`
**File:** `p4_multi_agent/tools/compliance_search.py`

Simulasi RAG (Retrieval-Augmented Generation) dari Project 2. Dalam
implementasi nyata, ini akan query vector database (ChromaDB/FAISS).
Di sini menggunakan dict in-memory dengan 5 kategori produk:

| Keyword       | HS Code | MFN Rate | EVFTA Rate |
|---------------|---------|----------|------------|
| textile       | 5208    | 12.0%    | 9.6%       |
| electronics   | 8471    | 0.0%     | 0.0%       |
| footwear      | 6404    | 17.0%    | ~7.5%      |
| pharmaceutical| 3004    | 0.0%     | 0.0%       |
| furniture     | 9403    | 2.7%     | 0.0%       |

Input: query string (nama produk atau HS code).
Output: JSON string berisi array record yang cocok.

Kalau tidak ada keyword yang match, mengembalikan **semua** record
sebagai fallback — supaya agent tetap punya data untuk diproses.

---

### `web_search`
**File:** `p4_multi_agent/tools/web_search.py`

Wrapper untuk Tavily Search API. Mengembalikan 3 hasil teratas dari web.

Jika `TAVILY_API_KEY` tidak di-set, tool ini mengembalikan **stub
string** yang berisi data simulasi — sistem tetap berjalan tanpa crash.

```python
if _tavily is None:
    return "[WEB SEARCH STUB] ..."  # data simulasi
```

---

### `calculate_tariff_impact`
**File:** `p4_multi_agent/tools/analysis_tools.py`

Kalkulasi murni Python — tidak ada LLM call di sini. Deterministik dan
bisa diverifikasi manual.

```
Input:  base_price=5.0, tariff_rate=12.0, volume=10000, currency="EUR"
Output:
  Base price per unit:      EUR 5.00
  Tariff rate:              12.0%
  Duty per unit:            EUR 0.60
  Landed cost per unit:     EUR 5.60
  Volume:                   10,000 units
  Total duty cost:          EUR 6,000.00
  Total landed cost:        EUR 56,000.00
  Duty as % of total cost:  10.7%
```

---

### `summarize_research`
**File:** `p4_multi_agent/tools/analysis_tools.py`

Ekstraksi deterministik — bukan LLM. Memecah teks research menjadi
kalimat-kalimat, memberi skor berdasarkan kehadiran kata kunci compliance
(`tariff`, `rate`, `duty`, `regulation`, `certificate`, dll.), lalu
mengambil N kalimat dengan skor tertinggi sebagai bullet points.

---

## Setiap Agent

### Supervisor
**File:** `p4_multi_agent/agents/supervisor.py`

**Model:** `fast_llm` (gpt-4o-mini, temperature=0)

**Tugasnya:** membaca state, memutuskan siapa yang dipanggil berikutnya.

**Cara kerjanya:**

1. Membaca 3 boolean dari state: `has_research`, `has_analysis`, `has_report`
2. Membangun prompt yang berisi system prompt + status ketiga field itu
3. Mengirim ke LLM via `.with_structured_output(RouteDecision)`
4. LLM mengembalikan objek `RouteDecision` dengan field `next` dan `reason`
5. Mengembalikan `{"next": "...", "iteration_count": iteration + 1}`

**Kenapa `with_structured_output`?**

Alternatifnya adalah parsing free-text seperti `if "Researcher" in response`.
Fragile — kalau model mengembalikan "go to Researcher" alih-alih "Researcher",
parsing gagal dan supervisor looping. Dengan Pydantic schema, LLM dipaksa
mengembalikan salah satu dari 4 nilai yang valid (`Literal` type).

**Routing rules (urutan prioritas):**
```
research_findings kosong          -> Researcher
analysis_results kosong           -> Analyst
final_report kosong               -> Writer
final_report ada                  -> FINISH
iteration_count > 8 & belum FINISH -> Writer (anti-loop emergency)
```

**Apa yang TIDAK dilakukan Supervisor:** Supervisor tidak melihat isi
`research_findings` atau `analysis_results` — hanya tahu apakah field itu
kosong atau tidak. Ini trade-off: routing jadi simpel dan murah, tapi
Supervisor tidak bisa mendeteksi bahwa research-nya buruk.

---

### Researcher
**File:** `p4_multi_agent/agents/researcher.py`

**Model:** `fast_llm` (gpt-4o-mini, temperature=0)
**Tools:** `search_compliance_docs`, `web_search`
**Pattern:** `create_react_agent` (ReAct loop)

**Cara kerjanya:**

`create_react_agent` membuat sub-graph ReAct: model berpikir → pilih tool →
eksekusi tool → lihat hasil → berpikir lagi → sampai model memutuskan selesai.

```
[Researcher menerima query]
    |
    v
Thought: "Saya perlu cari data tarif tekstil Vietnam"
    |
    v
Action: search_compliance_docs("textile Vietnam tariff")
    |
    v
Observation: {"hs_code": "5208", "eu_tariff_rate_pct": 12.0, ...}
    |
    v
Thought: "Ada data dasar. Perlu cek update terbaru di web"
    |
    v
Action: web_search("EU textile tariff Vietnam 2024 update")
    |
    v
Observation: "[hasil web / stub]"
    |
    v
Final Answer: "PRODUCT: Woven fabrics... EU TARIFF RATE: 12.0%..."
```

**Output yang ditulis ke state:** `research_findings` (string terformat)

**Input yang dibaca dari state:** hanya `query`

---

### Analyst
**File:** `p4_multi_agent/agents/analyst.py`

**Model:** `fast_llm` (gpt-4o-mini, temperature=0)
**Tools:** `calculate_tariff_impact`, `summarize_research`
**Pattern:** `create_react_agent` (ReAct loop)

**Cara kerjanya:**

Menerima `research_findings` + `query` dari state (digabung jadi satu
HumanMessage). ReAct loop berjalan:

```
[Analyst menerima research + query]
    |
    v
Thought: "Ada tariff_rate 12.0%, base_price 5 EUR, volume 10000"
    |
    v
Action: calculate_tariff_impact(base_price=5.0, tariff_rate=12.0, volume=10000)
    |
    v
Observation: "Total duty: EUR 6,000.00 | Landed cost: EUR 56,000.00"
    |
    v
Action: summarize_research(research_text="PRODUCT: Woven fabrics...")
    |
    v
Observation: "• EVFTA preferential rate applies with EUR.1 certificate..."
    |
    v
Final Answer: "FINANCIAL ANALYSIS: ... KEY COMPLIANCE POINTS: ..."
```

**Penting:** Prompt Analyst mewajibkan kedua tool dipanggil. Kalau tidak
ada angka eksplisit di research, Analyst diminta pakai default
(volume=1000, base_price=10.0) dan mencatatnya.

**Output yang ditulis ke state:** `analysis_results`

**Input yang dibaca dari state:** `research_findings`, `query`

---

### Writer
**File:** `p4_multi_agent/agents/writer.py`

**Model:** `writer_llm` (gpt-4o-mini, temperature=0.3)
**Tools:** tidak ada
**Pattern:** direct `llm.invoke()` — bukan ReAct

**Kenapa tidak pakai `create_react_agent`?**

Writer tidak butuh tools — tugasnya murni sintesis teks. Menambahkan
ReAct overhead (extra LLM call untuk "pilih tool") tidak ada nilainya.
Direct invoke lebih cepat dan lebih murah.

**Cara kerjanya:**

Membangun satu prompt besar yang berisi system prompt + query + research +
analysis, lalu kirim sekali ke LLM:

```
[Writer menerima research + analysis + query]
    |
    v
LLM invoke (satu call, tidak ada loop)
    |
    v
Output: laporan terstruktur dengan 3 seksi wajib:
  ## Introduction     <- 2 paragraf konteks
  ## Key Findings     <- 5 bullet points
  ## Recommendation   <- 3-4 action items bernomor
```

**Output yang ditulis ke state:** `final_report`

**Input yang dibaca dari state:** `research_findings`, `analysis_results`, `query`

---

## Graph & Routing

**File:** `p4_multi_agent/graph.py`

```python
workflow = StateGraph(AgentState)

# Daftarkan 4 node
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst",    analyst_node)
workflow.add_node("Writer",     writer_node)

# Entry point selalu Supervisor
workflow.set_entry_point("supervisor")

# Edge bercabang dari Supervisor berdasarkan state["next"]
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "Researcher": "Researcher",
        "Analyst":    "Analyst",
        "Writer":     "Writer",
        "FINISH":     END,         # graph berhenti
    }
)

# Semua sub-agent kembali ke Supervisor setelah selesai
workflow.add_edge("Researcher", "supervisor")
workflow.add_edge("Analyst",    "supervisor")
workflow.add_edge("Writer",     "supervisor")
```

**Recursion limit:** `graph.invoke(state, config={"recursion_limit": 15})`
LangGraph akan raise `RecursionError` setelah 15 step. Ini safety net
terakhir — Supervisor punya mekanisme anti-loop sendiri di iteration_count > 8.

---

## Alur End-to-End

Query contoh nyata dari run terakhir:
> *"Research EU new tariff implications for electronics products imported
> from China in 2024, calculate cost impact for 5000 units at 50 EUR base
> price, and write an executive summary"*

```
Step 1: graph.invoke() dipanggil
        State awal: semua field kosong, iteration_count=0

Step 2: SUPERVISOR (iter=1)
        Baca state: has_research=False, has_analysis=False, has_report=False
        LLM memutuskan: next="Researcher"
        Reason: "Research findings are empty"
        State berubah: next="Researcher", iteration_count=1

Step 3: RESEARCHER
        Tool call #1: search_compliance_docs("electronics China tariff")
          -> dapat record HS 8471: tariff 0.0%, ITA zero-duty
        Tool call #2: web_search("EU electronics tariff China 2024")
          -> dapat stub (Tavily tidak dikonfigurasi)
        Sintesis -> research_findings terisi
        State berubah: research_findings="PRODUCT: electronics (HS 8471)..."

Step 4: SUPERVISOR (iter=2)
        Baca state: has_research=True, has_analysis=False, has_report=False
        LLM memutuskan: next="Analyst"
        Reason: "Research complete, need calculations"
        State berubah: next="Analyst", iteration_count=2

Step 5: ANALYST
        Baca research_findings dari state
        Tool call #1: calculate_tariff_impact(base_price=50, tariff_rate=0.0, volume=5000)
          -> Total duty: EUR 0.00, Total landed cost: EUR 250,000.00
        Tool call #2: summarize_research(research_findings)
          -> 5 bullet points compliance
        Sintesis -> analysis_results terisi
        State berubah: analysis_results="FINANCIAL ANALYSIS:..."

Step 6: SUPERVISOR (iter=3)
        Baca state: has_research=True, has_analysis=True, has_report=False
        LLM memutuskan: next="Writer"
        Reason: "Both research and analysis present, need final report"
        State berubah: next="Writer", iteration_count=3

Step 7: WRITER
        Baca research_findings + analysis_results + query dari state
        Satu LLM call -> laporan dengan 3 seksi
        State berubah: final_report="## Introduction..."

Step 8: SUPERVISOR (iter=4)
        Baca state: has_research=True, has_analysis=True, has_report=True
        LLM memutuskan: next="FINISH"
        State berubah: next="FINISH", iteration_count=4

Step 9: Graph berhenti (FINISH -> END)
        graph.invoke() mengembalikan state final

Total: 4 iterasi Supervisor, 2 handoff antar agent, wall time ~38 detik
```

### Diagram State Transitions

```
                   [START]
                      |
              state["next"]=""
                      |
               +------v------+
               | SUPERVISOR  |  iter=1 -> next="Researcher"
               +------+------+
                      |
              +-------v--------+
              |   RESEARCHER   |  tool: search_compliance_docs
              |                |  tool: web_search
              +-------+--------+
                      |
               +------v------+
               | SUPERVISOR  |  iter=2 -> next="Analyst"
               +------+------+
                      |
               +------v------+
               |   ANALYST   |  tool: calculate_tariff_impact
               |             |  tool: summarize_research
               +------+------+
                      |
               +------v------+
               | SUPERVISOR  |  iter=3 -> next="Writer"
               +------+------+
                      |
               +------v------+
               |   WRITER    |  direct LLM invoke (no tools)
               +------+------+
                      |
               +------v------+
               | SUPERVISOR  |  iter=4 -> next="FINISH"
               +------+------+
                      |
                    [END]
```

---

## Cara Menjalankan

### Setup awal (sekali saja)

```powershell
# 1. Buat file .env
cd A:\Web\agentic-llm
copy .env.example .env
# Edit .env, isi SUMOPOD_API_KEY dengan key kamu

# 2. Pastikan dependencies terinstall
pip install langgraph langchain langchain-anthropic langchain-community langchain-openai python-dotenv pydantic
```

### Jalankan

```powershell
cd A:\Web\agentic-llm
$env:PYTHONIOENCODING="utf-8"
python -m p4_multi_agent.main
```

Query custom (tambahkan sebagai argument):

```powershell
python -m p4_multi_agent.main "riset implikasi tarif EU untuk produk tekstil dari Vietnam, hitung dampak biaya 10000 unit harga EUR 5, tulis ringkasan eksekutif"
```

### Produk yang didukung mock database

`textile`, `electronics`, `footwear`, `pharmaceutical`, `furniture`

Query yang mengandung kata lain akan mengembalikan semua 5 produk sebagai
fallback.

---

## Kelemahan Nyata & Trade-off

Ini bukan teori dari dokumentasi — ini yang ditemukan saat testing langsung.

### Kelemahan 1: Latency berlipat ganda

| Versi        | API Calls | Wall Time |
|--------------|-----------|-----------|
| Single-agent | ~3 calls  | ~8-12s    |
| Multi-agent  | ~9 calls  | ~35-40s   |

Setiap Supervisor call = satu round-trip ke API (~2s). Ditambah ReAct
loop masing-masing agent (2-3 call per agent). Total 9+ calls untuk
query yang sebenarnya bisa diselesaikan single-agent dalam 3 calls.

### Kelemahan 2: State passing yang lossy

Supervisor hanya tahu `has_research = True/False` — tidak membaca **isi**
`research_findings`. Kalau Researcher menghasilkan data yang ambigu atau
salah, Supervisor tidak tahu. Analyst kemudian meneruskan angka yang
salah ke Writer, dan Writer dengan percaya diri melaporkannya.

Di single-agent, satu model memegang semua konteks — jauh lebih mudah
mendeteksi inkonsistensi internal.

### Kelemahan 3: Debugging multi-hop susah

Ketika output final salah, pertanyaannya adalah: salah di mana?
- Research yang buruk?
- Analyst salah baca angka?
- Writer salah sintesis?
- Supervisor salah routing?

Tanpa logging eksplisit di tiap node (seperti `print("[ANALYST] Done...")`)
atau tool seperti LangSmith, kamu hanya dapat laporan akhir yang salah
tanpa tahu step mana yang break.

Di single-agent: satu chain-of-thought, langsung kelihatan di mana
reasoningnya keliru.

---

### Kapan Single-Agent Lebih Baik?

| Skenario | Pilihan Terbaik | Alasan |
|----------|-----------------|--------|
| Query sederhana ("berapa tarif HS 8471?") | Single-agent | Multi-agent 4x lebih lambat, output sama |
| Latency penting (chatbot real-time) | Single-agent | 38s tidak acceptable untuk UX |
| Budget terbatas | Single-agent | Multi-agent 3x lebih banyak token |
| Sedang development/iterasi prompt | Single-agent | Debug jauh lebih mudah |
| Query kompleks multi-domain | Multi-agent | Spesialisasi tool tiap agent masuk akal |
| Scale besar (ratusan query parallel) | Multi-agent | Bisa assign model berbeda per tugas |

**Kesimpulan jujur:** Untuk project ini, single-agent dengan semua 4 tools
akan menghasilkan output yang setara dengan waktu 3x lebih cepat dan biaya
2-3x lebih murah. Multi-agent worth it hanya kalau:
1. Setiap agent perlu model yang berbeda (misal: Researcher pakai model
   dengan browsing, Writer pakai model dengan fine-tuning laporan)
2. Agent bisa berjalan paralel (di sini semua sequential)
3. Tim dev yang berbeda maintain agent yang berbeda
