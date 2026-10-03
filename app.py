"""Лаб. 3.1 · Streamlit-дашборд · Вариант 8: мониторинг отелей · Момунов Акыл, ПИ-2-23
Запуск: streamlit run app.py  (рядом должны лежать hotel_clean.csv и папка .streamlit)"""
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Hotel Analytics", page_icon="🏨", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.hero{padding:26px 32px;border-radius:22px;background:linear-gradient(120deg,#4f46e5,#8b5cf6 50%,#ec4899);margin-bottom:18px;box-shadow:0 12px 40px rgba(139,92,246,.35)}
.hero h1{margin:0;font-size:2.1rem;font-weight:800;color:#fff}.hero p{margin:6px 0 0;color:#ffffffd0}
div[data-testid="stMetric"]{background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.09);padding:16px 18px;border-radius:16px}
.stTabs [data-baseweb="tab-list"]{gap:6px}
.stTabs [data-baseweb="tab"]{background:rgba(255,255,255,.04);border-radius:12px;padding:8px 16px}
.stTabs [aria-selected="true"]{background:linear-gradient(90deg,#4f46e5,#8b5cf6)}
.stTabs [data-baseweb="tab-highlight"]{display:none}
.ins{background:rgba(255,255,255,.045);border-left:4px solid #8b5cf6;padding:12px 16px;border-radius:12px;margin-bottom:10px}
</style>""", unsafe_allow_html=True)

PAL = ["#8b5cf6", "#22d3ee", "#f472b6", "#fbbf24", "#34d399", "#60a5fa", "#fb7185", "#a3e635"]
MONTHS = ["", "январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
GEO = {"Silk Road Palace": (42.875, 74.60), "Ala-Too Business": (42.88, 74.57), "Manas Inn": (42.86, 74.55),
       "Tian Shan Hostel": (42.85, 74.62), "Issyk-Kul Grand Resort": (42.65, 77.08), "Blue Lake Resort": (42.64, 77.06),
       "Pearl Beach": (42.62, 77.19), "Karakol Ski Lodge": (42.49, 78.39), "Altyn Arashan Camp": (42.35, 78.50),
       "Osh Heritage": (40.53, 72.80), "Sulaiman Hotel": (40.52, 72.79), "Naryn Mountain Inn": (41.43, 75.99)}

def style(fig, h=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h,
                      margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Inter"), colorway=PAL, legend_title_text="")
    return fig

@st.cache_data
def load(src):
    d = pd.read_csv(src, parse_dates=["date"])
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    d["lat"] = d["hotel_name"].map(lambda n: GEO[n][0]); d["lon"] = d["hotel_name"].map(lambda n: GEO[n][1])
    return d

try: df = load("hotel_clean.csv")
except FileNotFoundError:
    up = st.sidebar.file_uploader("Загрузите hotel_clean.csv", type="csv")
    if up is None: st.info("Положите hotel_clean.csv рядом с app.py."); st.stop()
    df = load(up)

REG = {"stars": "Звёздность", "rooms_total": "Номеров", "occupancy_rate": "Загрузка", "avg_rating": "Рейтинг",
       "competitor_adr_som": "Чек конкурентов", "marketing_spend_som": "Маркетинг", "is_weekend": "Выходной", "is_holiday": "Праздник"}
PRB = ["avg_rating", "adr_k", "marketing_k", "temperature_c", "is_weekend", "is_holiday", "stars"]
CLF = {"occupancy_rate": "Загрузка", "adr_som": "Чек", "avg_rating": "Рейтинг", "avg_stay_nights": "Ночей", "cancel_rate": "Отмены", "marketing_spend_som": "Маркетинг"}

@st.cache_resource
def train_adr(d):
    X, y = d[list(REG)], d["adr_som"]
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=.2, random_state=42)
    ols = sm.OLS(ya, sm.add_constant(Xa)).fit()
    rf = RandomForestRegressor(120, max_depth=14, random_state=42, n_jobs=-1).fit(Xa, ya)
    sc = {"МНК": (r2_score(yb, ols.predict(sm.add_constant(Xb, has_constant="add"))), 0), "Random Forest": (r2_score(yb, rf.predict(Xb)), 0)}
    sc["МНК"] = (sc["МНК"][0], np.sqrt(mean_squared_error(yb, ols.predict(sm.add_constant(Xb, has_constant="add")))))
    sc["Random Forest"] = (sc["Random Forest"][0], np.sqrt(mean_squared_error(yb, rf.predict(Xb))))
    return ols, rf, sc

@st.cache_resource
def train_prob(d):
    z = d.assign(adr_k=d.adr_som / 1000, marketing_k=d.marketing_spend_som / 1000)
    X = sm.add_constant(z[PRB]); y = (z.occupancy_rate >= .7).astype(int)
    return sm.Probit(y, X).fit(disp=0), X.mean(), y.mean()

@st.cache_resource
def train_fc(d):
    t0 = d.date.min()
    ft = lambda hid, dt: pd.DataFrame({"h": hid, "doy": dt.dayofyear, "dow": dt.dayofweek, "t": (dt - t0).days})
    X, y = ft(d.hotel_id.values, pd.DatetimeIndex(d.date)), d.occupancy_rate.values
    tr = (d.date <= d.date.max() - pd.Timedelta(days=90)).values
    mk = lambda: RandomForestRegressor(150, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rmse = np.sqrt(mean_squared_error(y[~tr], mk().fit(X[tr], y[tr]).predict(X[~tr])))
    return mk().fit(X, y), rmse, ft

ols, rf, adr_sc = train_adr(df); prob_m, x_mean, share = train_prob(df); fc_m, fc_rmse, fc_ft = train_fc(df)

st.markdown(f"""<div class="hero"><h1>🏨 Hotel Analytics · Кыргызстан</h1>
<p>Мониторинг эффективности отелей · данные {df.date.min():%d.%m.%Y} – {df.date.max():%d.%m.%Y} · Момунов Акыл, ПИ-2-23</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Фильтры")
    cities = st.multiselect("Город", sorted(df.city.unique()), default=sorted(df.city.unique()))
    av = sorted(df[df.city.isin(cities)].hotel_name.unique())
    hotels = st.multiselect("Отели", av, default=av)
    d0, d1 = df.date.min().date(), df.date.max().date()
    period = st.date_input("Период", (d0, d1), min_value=d0, max_value=d1)
if len(period) != 2 or not hotels: st.warning("Выберите отели и период."); st.stop()
base = df[df.hotel_name.isin(hotels)]
f = base[base.date.between(pd.Timestamp(period[0]), pd.Timestamp(period[1]))]
if f.empty: st.warning("Нет данных."); st.stop()
span = pd.Timestamp(period[1]) - pd.Timestamp(period[0])
prev = base[base.date.between(pd.Timestamp(period[0]) - span - pd.Timedelta(days=1), pd.Timestamp(period[0]) - pd.Timedelta(days=1))]

tabs = st.tabs(["📊 Обзор", "🗓️ Сезонность", "🧩 Сегменты", "🔮 Прогноз загрузки", "💰 Калькулятор чека", "🎯 Probit"])

# ---------------------------------------------------------------- Обзор
with tabs[0]:
    def dl(c, p, pct=True): return None if prev.empty or not p else (f"{(c / p - 1):+.1%}" if pct else f"{(c - p) * 100:+.1f} п.п.")
    cur = (f.occupancy_rate.mean(), f.adr_som.mean(), f.revpar_som.mean(), f.avg_rating.mean(), f.revenue_som.sum())
    pv = (prev.occupancy_rate.mean(), prev.adr_som.mean(), prev.revpar_som.mean(), prev.avg_rating.mean(), prev.revenue_som.sum()) if not prev.empty else (0,) * 5
    k = st.columns(5)
    k[0].metric("Загрузка", f"{cur[0]:.1%}", dl(cur[0], pv[0], False)); k[1].metric("Средний чек", f"{cur[1]:,.0f} сом", dl(cur[1], pv[1]))
    k[2].metric("RevPAR", f"{cur[2]:,.0f} сом", dl(cur[2], pv[2])); k[3].metric("Рейтинг", f"{cur[3]:.2f} ★", dl(cur[3], pv[3]))
    k[4].metric("Выручка", f"{cur[4] / 1e6:,.0f} млн", dl(cur[4], pv[4]))
    st.caption("Изменение — относительно предыдущего периода такой же длины.")

    ya = base[base.date > base.date.max() - pd.Timedelta(days=365)].adr_som.mean(); yb = base[(base.date <= base.date.max() - pd.Timedelta(days=365)) & (base.date > base.date.max() - pd.Timedelta(days=730))].adr_som.mean()
    g = f.groupby("hotel_name").agg(rv=("revpar_som", "mean"), occ=("occupancy_rate", "mean"), rt=("avg_rating", "mean"))
    wk = f.groupby("is_weekend").occupancy_rate.mean(); mm = f.groupby(f.date.dt.month).occupancy_rate.mean()
    ins = [f"🏆 Лидер по RevPAR — <b>{g.rv.idxmax()}</b> ({g.rv.max():,.0f} сом на номер в сутки).",
           f"🌞 Пик спроса — <b>{MONTHS[mm.idxmax()]}</b> ({mm.max():.0%}), провал — <b>{MONTHS[mm.idxmin()]}</b> ({mm.min():.0%}).",
           f"📈 Средний чек за последние 12 мес. вырос на <b>{(ya / yb - 1):.1%}</b> к предыдущему году." if yb == yb and yb else "",
           f"🗓️ В выходные загрузка {'выше' if wk.get(1, 0) > wk.get(0, 0) else 'ниже'} на <b>{abs(wk.get(1, 0) - wk.get(0, 0)) * 100:.1f} п.п.</b>",
           f"⭐ Корреляция рейтинга и загрузки: <b>{f.avg_rating.corr(f.occupancy_rate):.2f}</b>."]
    c1, c2 = st.columns([1.15, 1])
    with c1:
        fig = px.scatter_map(g.reset_index().assign(lat=lambda x: x.hotel_name.map(lambda n: GEO[n][0]), lon=lambda x: x.hotel_name.map(lambda n: GEO[n][1])),
                             lat="lat", lon="lon", size="rv", color="occ", hover_name="hotel_name", hover_data={"rv": ":,.0f", "occ": ":.0%", "lat": False, "lon": False},
                             color_continuous_scale="Turbo", size_max=32, zoom=5.3, center=dict(lat=41.9, lon=75.6), map_style="carto-darkmatter",
                             labels={"rv": "RevPAR", "occ": "Загрузка"})
        st.plotly_chart(style(fig, 430), width="stretch")
    with c2:
        st.markdown("#### 💡 Инсайты")
        for t in ins:
            if t: st.markdown(f'<div class="ins">{t}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    m = f.groupby(["month", "city"]).revenue_som.sum().reset_index()
    c1.plotly_chart(style(px.area(m, x="month", y="revenue_som", color="city", title="Выручка по месяцам, сом")), width="stretch")
    c2.plotly_chart(style(px.treemap(f, path=["city", "hotel_name"], values="revenue_som", color="occupancy_rate", color_continuous_scale="Viridis", title="Структура выручки (цвет — загрузка)")), width="stretch")

# ---------------------------------------------------------------- Сезонность
with tabs[1]:
    hm = f.pivot_table(index="hotel_name", columns="month", values="occupancy_rate")
    fig = px.imshow(hm, aspect="auto", color_continuous_scale="Plasma", title="Тепловая карта загрузки: отель × месяц", labels=dict(color="Загрузка"))
    st.plotly_chart(style(fig, 420), width="stretch")
    c1, c2 = st.columns(2)
    dw = f.groupby([f.date.dt.dayofweek.map(dict(enumerate(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]))).rename("день"), "season"]).occupancy_rate.mean().reset_index()
    c1.plotly_chart(style(px.bar(dw, x="день", y="occupancy_rate", color="season", barmode="group", title="Загрузка по дням недели и сезонам",
                    category_orders={"день": ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]})), width="stretch")
    c2.plotly_chart(style(px.scatter(f.sample(min(3000, len(f)), random_state=1), x="occupancy_rate", y="adr_som", color="stars", opacity=.6,
                    trendline="ols", title="Средний чек vs загрузка", color_continuous_scale="Turbo")), width="stretch")
    ac = [f.groupby("date").occupancy_rate.mean().autocorr(l) for l in range(1, 61)]
    st.plotly_chart(style(px.bar(x=range(1, 61), y=ac, title="Автокорреляция загрузки (лаги 1–60 дней)", labels=dict(x="лаг, дней", y="ACF")), 280), width="stretch")

# ---------------------------------------------------------------- Сегменты
with tabs[2]:
    X = StandardScaler().fit_transform(df[list(CLF)])
    a, b = st.columns([1, 2])
    with a:
        ks = list(range(2, 9)); inr = [KMeans(k, n_init=3, random_state=42).fit(X).inertia_ for k in ks]
        st.plotly_chart(style(px.line(x=ks, y=inr, markers=True, title="Метод локтя", labels=dict(x="k", y="inertia")), 300), width="stretch")
        k = st.slider("Число кластеров k", 2, 8, 4)
    dc = df.assign(cluster=("Сегмент " + pd.Series(KMeans(k, n_init=10, random_state=42).fit_predict(X) + 1).astype(str)).values)
    with b:
        s = dc.sample(3000, random_state=1)
        st.plotly_chart(style(px.scatter_3d(s, x="occupancy_rate", y="adr_som", z="avg_rating", color="cluster", opacity=.7, title="Сегменты в 3D"), 430), width="stretch")
    pr = dc.groupby("cluster")[list(CLF)].mean(); z = (pr - pr.min()) / (pr.max() - pr.min() + 1e-9)
    fig = go.Figure()
    for cl in z.index: fig.add_trace(go.Scatterpolar(r=z.loc[cl].tolist() + [z.loc[cl].iloc[0]], theta=list(CLF.values()) + [list(CLF.values())[0]], fill="toself", name=cl, opacity=.6))
    c1, c2 = st.columns(2)
    c1.plotly_chart(style(fig.update_layout(title="Профили сегментов (радар)", polar=dict(bgcolor="rgba(0,0,0,0)")), 400), width="stretch")
    c2.plotly_chart(style(px.histogram(dc, x="hotel_name", color="cluster", barnorm="percent", title="Состав сегментов по отелям, %").update_xaxes(tickangle=40), 400), width="stretch")
    st.dataframe(pr.round(2).rename(columns=CLF).assign(наблюдений=dc.groupby("cluster").size()), width="stretch")

# ---------------------------------------------------------------- Прогноз загрузки
with tabs[3]:
    hn = st.selectbox("Отель", sorted(df.hotel_name.unique()), index=4)
    end = st.date_input("Прогноз до", pd.Timestamp("2026-12-31"), min_value=df.date.max().date() + pd.Timedelta(days=7), max_value=pd.Timestamp("2027-12-31"))
    hid = int(df[df.hotel_name == hn].hotel_id.iloc[0]); h = df[df.hotel_name == hn].sort_values("date")
    fd = pd.date_range(df.date.max() + pd.Timedelta(days=1), pd.Timestamp(end)); pred = fc_m.predict(fc_ft(hid, fd))
    hist = h.tail(150)
    fig = go.Figure()
    fig.add_scatter(x=hist.date, y=hist.occupancy_rate.rolling(7, min_periods=1).mean(), name="Факт (7-дн. среднее)", line=dict(color="#22d3ee", width=3))
    ps = pd.Series(pred, index=fd).rolling(7, min_periods=1).mean()
    fig.add_scatter(x=fd, y=ps + fc_rmse, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fd, y=(ps - fc_rmse).clip(0), fill="tonexty", fillcolor="rgba(139,92,246,.22)", line=dict(width=0), name="Доверит. коридор ±RMSE")
    fig.add_scatter(x=fd, y=ps, name="Прогноз", line=dict(color="#a78bfa", width=3, dash="dot"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(style(fig.update_layout(title=f"{hn}: загрузка номеров — факт и прогноз"), 440), width="stretch")
    c = st.columns(3)
    c[0].metric("Средняя прогн. загрузка", f"{pred.mean():.1%}"); c[1].metric("Пиковый день", f"{pd.Series(pred, index=fd).idxmax():%d.%m.%Y}")
    c[2].metric("Ошибка модели (RMSE, тест 90 дн.)", f"{fc_rmse:.3f}")
    st.caption("Модель Random Forest учитывает сезон, день недели, отель и тренд роста. Проверена на последних 90 днях данных.")

# ---------------------------------------------------------------- Калькулятор чека
with tabs[4]:
    c = st.columns(2)
    for col_, (n, (r2, rm)) in zip(c, adr_sc.items()): col_.metric(n, f"R² = {r2:.3f}", f"RMSE {rm:,.0f} сом", delta_color="off")
    mdl = st.radio("Модель", list(adr_sc), horizontal=True)
    a, b = st.columns(2); inp = {}
    with a:
        inp["stars"] = st.select_slider("Звёздность", [2, 3, 4, 5], 4); inp["rooms_total"] = st.slider("Номеров", 30, 150, 80)
        inp["occupancy_rate"] = st.slider("Загрузка", .1, 1., .65, .01); inp["avg_rating"] = st.slider("Рейтинг", 2.5, 5., 4.2, .1)
    with b:
        inp["competitor_adr_som"] = st.slider("Чек конкурентов, сом", 1500, 16000, 6500, 100); inp["marketing_spend_som"] = st.slider("Маркетинг, сом/день", 500, 6000, 2000, 50)
        inp["is_weekend"] = int(st.toggle("Выходной")); inp["is_holiday"] = int(st.toggle("Праздник"))
    P = lambda r: float(ols.predict(sm.add_constant(r[list(REG)], has_constant="add")).iloc[0]) if mdl == "МНК" else float(rf.predict(r[list(REG)])[0])
    row = pd.DataFrame([inp]); p = P(row)
    st.success(f"Рекомендуемый средний чек: **{p:,.0f} сом** ({p / inp['competitor_adr_som'] - 1:+.0%} к конкурентам)")
    occs = np.linspace(.1, 1, 25); cur = pd.DataFrame([{**inp, "occupancy_rate": o} for o in occs])
    ys = [P(cur.iloc[[i]]) for i in range(len(cur))]
    fig = px.line(x=occs, y=ys, labels=dict(x="Загрузка", y="Чек, сом"), title="What-if: как чек зависит от загрузки")
    fig.add_scatter(x=[inp["occupancy_rate"]], y=[p], mode="markers", marker=dict(size=14, color="#f472b6"), name="Ваш сценарий"); fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, 320), width="stretch")
    with st.expander("Коэффициенты МНК"):
        st.dataframe(pd.DataFrame({"коэф.": ols.params, "t": ols.tvalues, "p-value": ols.pvalues}).round(4)); st.caption(f"R² = {ols.rsquared:.3f}, F = {ols.fvalue:,.0f}")

# ---------------------------------------------------------------- Probit
with tabs[5]:
    st.write(f"Вероятность того, что загрузка отеля **≥ 70 %** (таких дней в данных {share:.0%}).")
    a, b = st.columns(2)
    with a:
        r = st.slider("Рейтинг", 2.5, 5., 4., .1, key="r"); ad = st.slider("Чек, сом", 1500, 16000, 6000, 100, key="a"); mk = st.slider("Маркетинг, сом/день", 500, 6000, 2000, 50, key="m")
    with b:
        tp = st.slider("Температура, °C", -20, 35, 20, key="t"); sr = st.select_slider("Звёзды", [2, 3, 4, 5], 4, key="s"); we = int(st.toggle("Выходной", key="w")); ho = int(st.toggle("Праздник", key="h"))
    pb = float(prob_m.predict(pd.DataFrame([[1, r, ad / 1000, mk / 1000, tp, we, ho, sr]], columns=["const"] + PRB)).iloc[0])
    g = go.Figure(go.Indicator(mode="gauge+number", value=pb * 100, number=dict(suffix="%"), title=dict(text="Вероятность высокой загрузки"),
                  gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#8b5cf6"), steps=[dict(range=[0, 40], color="#7f1d1d"), dict(range=[40, 70], color="#78350f"), dict(range=[70, 100], color="#14532d")])))
    c1, c2 = st.columns([1, 1.2]); c1.plotly_chart(style(g, 320), width="stretch")
    me = prob_m.get_margeff(at="mean").summary_frame()["dy/dx"].rename(index={"adr_k": "чек (+1 тыс. сом)", "marketing_k": "маркетинг (+1 тыс.)"})
    c2.plotly_chart(style(px.bar(me.sort_values(), orientation="h", title="Предельные эффекты (Δ вероятности)", labels=dict(value="Δ P", index="")), 320), width="stretch")
    st.caption(f"Для «среднего» объекта выборки: Φ({float(prob_m.params @ x_mean):.2f}) = {norm.cdf(float(prob_m.params @ x_mean)):.1%}")
