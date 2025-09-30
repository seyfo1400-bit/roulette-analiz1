# app.py
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import altair as alt

st.set_page_config(page_title="Roulette Profesyonel Analiz", layout="wide")
st.title("🎯 Roulette — Profesyonel Analiz & Tahmin (Markov + Recency + MC)")

#
# ------ Yardımcı fonksiyonlar ------
#
RED = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
BLACK = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}
ALL_NUMS = list(range(0,37))

def color_of(n):
    if n == 0: return "Yeşil"
    if n in RED: return "Kırmızı"
    if n in BLACK: return "Siyah"
    return "?"

def dozen_of(n):
    if 1 <= n <= 12: return "1 (1-12)"
    if 13 <= n <= 24: return "2 (13-24)"
    if 25 <= n <= 36: return "3 (25-36)"
    return "0"

def column_of(n):
    if n == 0: return "0"
    return f"{((n-1)%3)+1}"

def safe_normalize(arr):
    s = np.sum(arr)
    return arr / s if s > 0 else np.ones_like(arr) / len(arr)

#
# ------ UI: Input bölümü ------
#
st.sidebar.header("Veri Girişi")
st.sidebar.write("1) En kolay: sayı listeni yapıştır (virgül veya yeni satır ile ayır).")
st.sidebar.write("2) Alternatif: CSV/TXT yükle (tek sütun veya virgülle ayrılmış).")
txt = st.sidebar.text_area("Sayıları yapıştır (max 500):", height=160)
uploaded = st.sidebar.file_uploader("Veya CSV/TXT dosyası yükle", type=['csv','txt'])

# Opsiyonlar
st.sidebar.header("Model Ayarları")
max_len = st.sidebar.number_input("Maks veri (son N) kullan (max 500)", min_value=50, max_value=500, value=500, step=50)
half_life = st.sidebar.number_input("Recency half-life (ör. 100 daha yakın daha ağırlıklı)", min_value=1, max_value=1000, value=100)
ensemble_markov = st.sidebar.slider("Markov ağırlığı (alpha)", 0.0, 1.0, 0.6)
ensemble_freq   = st.sidebar.slider("Genel frekans ağırlığı (beta)", 0.0, 1.0, 0.2)
ensemble_recency = st.sidebar.slider("Recency ağırlığı (gamma)", 0.0, 1.0, 0.2)
mc_runs = st.sidebar.number_input("Monte Carlo simülasyon sayısı", min_value=1000, max_value=20000, value=8000, step=1000)

st.sidebar.write("Not: alpha+beta+gamma = 1'e normalizasyon yapılıyor.")

#
# ------ Parse input to list of ints ------
#
def parse_numbers(text, uploaded_file):
    nums = []
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            # flatten values
            vals = df.values.flatten()
            for v in vals:
                try:
                    n = int(str(v).strip())
                    nums.append(n)
                except:
                    pass
        except Exception:
            # try as plain text
            content = uploaded_file.getvalue().decode('utf-8')
            parts = [p for p in [x.strip() for x in content.replace("\n",",").split(",")] if p!='']
            for p in parts:
                if p.isdigit(): nums.append(int(p))
    if text:
        parts = [p for p in [x.strip() for x in text.replace("\n",",").split(",")] if p!='']
        for p in parts:
            if p.lstrip('-').isdigit():
                nums.append(int(p))
    return nums

numbers = parse_numbers(txt, uploaded)
if len(numbers) == 0:
    st.info("Lütfen sayı verisi girin (paste veya yükle). Örnek: 8,16,33,10,...")
    st.stop()

# Keep only valid roulette numbers 0-36 and last max_len
numbers = [n for n in numbers if 0 <= n <= 36]
if len(numbers) == 0:
    st.error("Girdiğiniz veriler içinde 0-36 arası sayı yok veya format hatası.")
    st.stop()

numbers = numbers[-int(max_len):]  # son N
n_len = len(numbers)

# DataFrame oluştur
df = pd.DataFrame({"index": range(1, n_len+1), "number": numbers})
df["color"] = df["number"].apply(color_of)
df["dozen"] = df["number"].apply(dozen_of)
df["column"] = df["number"].apply(column_of)
df["odd_even"] = df["number"].apply(lambda x: "Çift" if x!=0 and x%2==0 else ("Tek" if x!=0 else "0"))

#
# ------ Sol üst: genel istatistikler ------
#
st.subheader("📊 Genel İstatistik (son {} spin)".format(n_len))
col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam spin", n_len)
col2.metric("Kırmızı", int((df["color"]=="Kırmızı").sum()))
col3.metric("Siyah", int((df["color"]=="Siyah").sum()))
col4.metric("Yeşil (0)", int((df["color"]=="Yeşil").sum()))

# Görselleştirme
with st.expander("Detaylı dağılım grafikleri"):
    st.write("Renk dağılımı")
    cdata = df["color"].value_counts().reset_index()
    cdata.columns = ["Renk","Adet"]
    st.altair_chart(alt.Chart(cdata).mark_bar().encode(x='Renk', y='Adet'), use_container_width=True)

    st.write("Düzine dağılımı")
    ddata = df["dozen"].value_counts().reset_index()
    ddata.columns = ["Düzine","Adet"]
    st.altair_chart(alt.Chart(ddata).mark_bar().encode(x='Düzine', y='Adet'), use_container_width=True)

#
# ------ Hedef sayı seç ve analiz ------
#
st.sidebar.header("Analiz: Hedef sayı")
target = st.sidebar.number_input("Hangi sayı için analiz (target)", min_value=0, max_value=36, value=13)
top_k = st.sidebar.number_input("En çok gösterilecek sonuç sayısı (top k)", min_value=1, max_value=10, value=6)

# compute immediate successors of target
successors = []
for i in range(len(numbers)-1):
    if numbers[i] == target:
        successors.append(numbers[i+1])

succ_count = Counter(successors)
succ_total = sum(succ_count.values())

st.subheader(f"🔎 Analiz: '{target}' sayısından sonra gelenler (top {top_k})")
if succ_total == 0:
    st.warning(f"Veride '{target}' sayısından sonra gelen hiç sayı bulunamadı.")
else:
    succ_df = pd.DataFrame(succ_count.most_common(top_k), columns=["Sayı","Adet"])
    succ_df["Yüzde"] = succ_df["Adet"] / succ_total * 100
    succ_df["Renk"] = succ_df["Sayı"].apply(color_of)
    succ_df["Bölge"] = succ_df["Sayı"].apply(dozen_of)
    st.table(succ_df.style.hide_index())

    st.write("Renk dağılımı (target sonrası)")
    rc = pd.Series([color_of(x) for x in successors]).value_counts().reset_index()
    rc.columns = ["Renk","Adet"]
    st.altair_chart(alt.Chart(rc).mark_bar().encode(x='Renk', y='Adet'), use_container_width=True)

    st.write("Bölge dağılımı (target sonrası)")
    bc = pd.Series([dozen_of(x) for x in successors]).value_counts().reset_index()
    bc.columns = ["Bölge","Adet"]
    st.altair_chart(alt.Chart(bc).mark_bar().encode(x='Bölge', y='Adet'), use_container_width=True)

#
# ------ Markov transition matrix (counts & probabilities) ------
#
# transitions from i -> i+1 for entire series
pairs = []
for i in range(len(numbers)-1):
    pairs.append((numbers[i], numbers[i+1]))

trans_counts = pd.DataFrame(0, index=ALL_NUMS, columns=ALL_NUMS)
for a,b in pairs:
    trans_counts.loc[a, b] += 1

# row-normalize to probabilities (P(next|current))
trans_probs = trans_counts.div(trans_counts.sum(axis=1).replace(0,1), axis=0)

# show top successors for target from markov
markov_top = trans_probs.loc[target].sort_values(ascending=False).head(top_k)
m_df = pd.DataFrame({"Sayı": markov_top.index.astype(int), "P_markov": markov_top.values})
m_df["Renk"] = m_df["Sayı"].apply(color_of)
st.subheader(f"📈 Markov Model: P(next|{target}) — top {top_k}")
st.table(m_df.style.hide_index())

#
# ------ Recency-weighted transition matrix ------
#
# weights for each transition (index i corresponds to transition numbers[i] -> numbers[i+1])
m = len(pairs)
if m > 0:
    # exponential weight with half-life
    lam = np.log(2) / float(max(1, half_life))
    # positions 0..m-1 -> older to newer; assign weight = exp(-lam*(m-1-pos)) so newest has highest weight 1
    positions = np.arange(m)
    weights = np.exp(-lam * (m-1 - positions))  # newest gets largest weight
    # accumulate weighted counts
    trans_counts_w = pd.DataFrame(0.0, index=ALL_NUMS, columns=ALL_NUMS)
    for idx, (a,b) in enumerate(pairs):
        trans_counts_w.loc[a,b] += weights[idx]
    trans_probs_w = trans_counts_w.div(trans_counts_w.sum(axis=1).replace(0.0, 1.0), axis=0)
else:
    trans_probs_w = trans_probs.copy()

recency_top = trans_probs_w.loc[target].sort_values(ascending=False).head(top_k)
r_df = pd.DataFrame({"Sayı": recency_top.index.astype(int), "P_recency": recency_top.values})
r_df["Renk"] = r_df["Sayı"].apply(color_of)
st.subheader(f"🕒 Recency-weighted Model (half-life={half_life}): P(next|{target}) — top {top_k}")
st.table(r_df.style.hide_index())

#
# ------ Frequency model (overall next distribution) ------
#
# overall frequency of numbers as they appear (or as "next" values)
nexts = [b for (_,b) in pairs]  # all observed next numbers
freq_counts = Counter(nexts)
freq_arr = np.array([freq_counts.get(i,0) for i in ALL_NUMS], dtype=float)
P_freq = safe_normalize(freq_arr)
freq_df = pd.DataFrame({"Sayı": ALL_NUMS, "P_freq": P_freq})
freq_top = freq_df.sort_values("P_freq", ascending=False).head(top_k)
st.subheader(f"📌 Genel Frekans Modeli (tüm next değerler): top {top_k}")
st.table(freq_top.style.hide_index())

#
# ------ Ensemble (alpha * markov + beta * freq + gamma * recency) ------
#
# get distributions for the 'target' row for each model
p_markov = trans_probs.loc[target].values if target in trans_probs.index else np.zeros(len(ALL_NUMS))
p_recency = trans_probs_w.loc[target].values if target in trans_probs_w.index else np.zeros(len(ALL_NUMS))
p_freq = P_freq

# Normalize ensemble weights
total_w = ensemble_markov + ensemble_freq + ensemble_recency
if total_w == 0:
    alpha, beta, gamma = 0.6, 0.2, 0.2
else:
    alpha = ensemble_markov / total_w
    beta  = ensemble_freq / total_w
    gamma = ensemble_recency / total_w

p_ensemble = alpha * p_markov + beta * p_freq + gamma * p_recency
p_ensemble = safe_normalize(p_ensemble)

ens_df = pd.DataFrame({"Sayı": ALL_NUMS, "P_ensemble": p_ensemble})
ens_top = ens_df.sort_values("P_ensemble", ascending=False).head(top_k)
ens_top["Renk"] = ens_top["Sayı"].apply(color_of)
ens_top["Bölge"] = ens_top["Sayı"].apply(dozen_of)
st.subheader("🔀 Ensemble Tahmin (alpha*Markov + beta*Freq + gamma*Recency)")
st.write(f"alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}")
st.table(ens_top.style.hide_index())

#
# ------ Monte Carlo simülasyonu (belirsizlik tahmini) ------
#
st.subheader("🔁 Monte Carlo Simülasyonu ile Güven aralığı (örnek)")
if mc_runs > 0:
    rng = np.random.default_rng(12345)
    sims = rng.choice(ALL_NUMS, size=mc_runs, p=p_ensemble)
    sim_counts = Counter(sims)
    sim_df = pd.DataFrame(sim_counts.most_common()).rename(columns={0:"Sayı", 1:"Adet"})
    sim_df["Yüzde"] = sim_df["Adet"] / mc_runs * 100
    sim_top = sim_df.head(top_k)
    st.table(sim_top.style.hide_index())
else:
    st.info("MC simülasyon sayısını 1'den büyük yapın.")

#
# ------ Ek: Son 20 / 50 trendleri ------
#
st.subheader("📈 Kısa dönem trendler")
for window in (20,50):
    if n_len >= 5 and n_len >= window:
        w = df.tail(window)
        st.write(f"Son {window} dağılım (Renk)")
        temp = w["color"].value_counts().reset_index()
        temp.columns = ["Renk","Adet"]
        st.altair_chart(alt.Chart(temp).mark_bar().encode(x='Renk', y='Adet'), use_container_width=True)
    else:
        st.write(f"Son {window}: yeterli veri yok (gerekli: {window})")

#
# ------ Son notlar ve indirilebilir sonuç ------
#
st.markdown("---")
st.markdown("**Notlar:**")
st.markdown("""
- Model bir *tahmin* sunar; rulet temelde rastgeledir — kesinlik garanti edilemez.  
- `Markov` = geçmişte bir sayıdan sonra ne geldiğinin doğrudan olasılıklarıdır.  
- `Recency` = daha yeni veriler daha fazla ağırlık alır (half-life parametresi ile kontrol).  
- `Freq` = genel olarak hangi sayının ne kadar "next" olduğunu gösterir.  
- Ensemble bunları birleştirir; MC ile belirsizliği görürsün.  
""")

st.download_button("CSV olarak indir: Ensemble tahmin tablosu", ens_df.to_csv(index=False), file_name="ensemble_predictions.csv", mime="text/csv")
